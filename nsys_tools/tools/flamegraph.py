"""
This tool is used to detect training-step boundaries and emit a kernel->NVTX flame graph.
Author: yezhengmaolove@gmail.com
"""

import argparse
import json
import re
import sys
from pathlib import Path

from ..utils.common import (
    get_rank,
    has_table,
    human_ns,
    open_db,
    require_kernel_table,
    truncate,
)
from ..utils.kernel import KERNEL_SQL
from ..utils.nvtx import NvtxIndex


# ---- tunables ----
NS_PER_MS = 1e6  # ns→ms divisor for the step-window table
MARKER_W_MIN = 20  # min width of the 'marker' column in the step table
MARKER_W_MAX = 60  # max width of the 'marker' column in the step table


# =============================================================================
# Flame-graph rendering (stack tree + interactive HTML)
# =============================================================================


class StackNode:
    __slots__ = ("name", "kind", "value", "solo_value", "count", "children")

    def __init__(self, name: str, kind: str):
        self.name = name
        self.kind = kind
        self.value = 0
        self.solo_value = 0
        self.count = 0  # number of activities (kernel launches / gaps) through here
        self.children: dict[tuple[str, str], "StackNode"] = {}

    def add_path(self, frames: list[tuple[str, str]], value: int, solo: int) -> None:
        self.value += value
        self.solo_value += solo
        self.count += 1
        cur = self
        for name, kind in frames:
            key = (name, kind)
            c = cur.children.get(key)
            if c is None:
                c = StackNode(name, kind)
                cur.children[key] = c
            c.value += value
            c.solo_value += solo
            c.count += 1
            cur = c

    def clone(self) -> "StackNode":
        """Deep copy — used to build the idle-attributed variant without mutating
        the pure-activity tree that drives the Sum/Solo charts."""
        n = StackNode(self.name, self.kind)
        n.value = self.value
        n.solo_value = self.solo_value
        n.count = self.count
        n.children = {k: c.clone() for k, c in self.children.items()}
        return n


def compute_solo_times(intervals: list[tuple[int, int]]) -> list[int]:
    """
    For each kernel i, how much of [start_i, end_i) had no other kernel running.
    """
    N = len(intervals)
    if N == 0:
        return []
    # Events sorted by (time, delta) with ends (-1) before starts (+1) at same t.
    events: list[tuple[int, int, int]] = []
    for i, (s, e) in enumerate(intervals):
        events.append((s, +1, i))
        events.append((e, -1, i))
    events.sort(key=lambda ev: (ev[0], ev[1]))

    solos = [0] * N
    active: set[int] = set()
    last_t: int | None = None
    for t, delta, idx_ in events:
        if last_t is not None and last_t < t and len(active) == 1:
            solos[next(iter(active))] += t - last_t
        if delta == +1:
            active.add(idx_)
        else:
            active.discard(idx_)
        last_t = t
    return solos


def frame_tag(name: str, kind: str) -> str:
    """Display label for a frame. Leaves of different kinds get distinct prefixes."""
    if kind == "gpu":
        return f"[gpu] {name}"
    if kind == "idle":
        return f"[idle] {name}"
    return name


def _html_esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _tree_to_json(root: StackNode) -> dict:
    """Convert the stack tree into a view-agnostic d3-flame-graph JSON.

    Each node carries the additive quantities (sum / solo GPU time, and count =
    number of activities); the client-side JS picks one as d.value per view, and
    derives overlap (= sum - solo) for the Sum view's color saturation.
    """

    def rec(node: StackNode) -> dict:
        label = frame_tag(node.name, node.kind) if node.kind != "root" else "all"
        out = {
            "name": label,
            "kind": node.kind,
            "sum": node.value,
            "solo": node.solo_value,
            "count": node.count,
        }
        if node.children:
            out["children"] = [
                rec(c) for c in sorted(node.children.values(), key=lambda n: -n.value)
            ]
        return out

    return rec(root)


# The HTML/CSS/JS template lives in nsys_tools/templates/flamegraph.html (kept out
# of this module so editors highlight it as HTML); placeholders __TITLE__/__DATA__
# plus the __KEY__s in LABELS below.
_TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent / "templates" / "flamegraph.html"
)
FLAMEGRAPH_HTML_TEMPLATE = _TEMPLATE_PATH.read_text()

# Display labels filled into the template's __KEY__ placeholders.
LABELS = {
    "LEGEND": (
        '\n  <span class="sw" style="background:hsl(210,45%,62%)"></span> NVTX '
        "(CPU-side semantic marker; not GPU work)<br>\n"
        '  <span class="sw" style="background:hsl(25,80%,58%)"></span> GPU kernel '
        "leaf (actual SM execution)<br>\n"
        '  <span class="sw" style="background:hsl(140,58%,46%)"></span> GPU memcpy '
        "(H2D/D2H/D2D copy) &nbsp; "
        '<span class="sw" style="background:hsl(275,52%,56%)"></span> GPU memset<br>\n'
        '  <span class="sw" style="background:hsl(320,85%,55%)"></span> GPU idle '
        "(no GPU activity, attributed to the NVTX scope enclosing the gap) — "
        "bright magenta so it is unmistakable; gray is reserved for the "
        "<i>overlap</i> encoding in the Sum view\n"
    ),
    "NOTE": (
        "\n  Four charts below share the same hierarchy. Identity per node: "
        "<b>sum = solo + overlap</b>.<br>\n"
        "  <b>Solo</b> (non-overlap) — widths are per-activity solo time, the "
        "portion that actually blocks the GPU clock.<br>\n"
        "  <b>Sum</b> (overlap) — widths are total GPU-activity time; color "
        "saturation encodes overlap (vivid = exposed, pale = mostly hidden behind "
        "other GPU work). Wide + pale here but narrow in Solo → good concurrency; "
        "wide in both → fully exposed, optimization target.<br>\n"
        "  <b>Count</b> — widths are the <i>number</i> of GPU activities (kernel "
        "launches). Compare with Sum: wide in Count but narrow in Sum → lots of "
        "tiny kernels (launch-bound), a fusion/CUDA-graph target.<br>\n"
        "  <b>Sum + idle</b> — the Sum chart plus GPU-idle gaps added as magenta "
        "&lt;idle&gt; leaves under the NVTX scope that enclosed each gap, so you "
        "see which phase the GPU stalled in. Idle appears in this chart only; the "
        "others are pure GPU activity.<br>\n"
        "  Tooltip shows absolute time (sum / solo / overlap%) and count / % of "
        "launches for every node.\n"
    ),
    "VIEW1_H2": (
        "Sum view — widths ∝ Σ GPU kernel duration; "
        '<span style="font-weight:normal;color:#555;">rect <b>color saturation</b> '
        "encodes overlap: vivid = mostly exposed (blocking GPU), pale = mostly "
        "hidden (overlapped with other work)</span>"
    ),
    "VIEW2_H2": (
        "Solo view — widths ∝ Σ per-kernel solo time (GPU active_count == 1; the "
        "portion that actually blocks the GPU clock)"
    ),
    "COUNT_H2": (
        "Count view — widths &prop; number of GPU activities (kernel launches + "
        "memcpy/memset); tooltip shows count and % of launches. Compare against "
        "Sum to spot many-tiny-kernel overhead (wide here, narrow in Sum)"
    ),
    "IDLE_H2": (
        "Sum + idle attribution — same as Sum, plus GPU-idle gaps added as "
        "magenta <span style=\"color:#c0c\">&lt;idle&gt;</span> leaves under the "
        "NVTX scope that enclosed each gap (which phase the GPU stalled in)"
    ),
    "V1NAME": "sum",
    "V2NAME": "solo",
    "DIFFNAME": "overlap ratio",
    "DIFFFORMULA": "(1 - solo/sum)",
}

DIFF_LABELS = {
    "LEGEND": (
        '\n  <span class="sw" style="background:hsl(2,75%,55%)"></span> grew '
        '(after − before is positive) &nbsp; '
        '<span class="sw" style="background:hsl(0,0%,92%)"></span> unchanged '
        '&nbsp; <span class="sw" style="background:hsl(212,75%,55%)"></span> '
        "shrank (after − before is negative)\n"
    ),
    "NOTE": (
        "\n  This is the same renderer and hierarchy as the normal flame graph. "
        "Widths are absolute leaf changes rolled up through the tree; colors "
        "encode signed <b>after − before</b>. Red grew, blue shrank.<br>\n"
        "  <b>Solo / Sum / Count / Sum + idle</b> have the same positions and "
        "interactions as the normal flame graph. Tooltips show the displayed "
        "change magnitude and signed net delta.\n"
    ),
    "VIEW1_H2": "Sum view — widths ∝ rolled-up |Δ GPU-activity time|; color = signed Δ",
    "VIEW2_H2": "Solo view — widths ∝ rolled-up |Δ per-activity solo time|; color = signed Δ",
    "COUNT_H2": "Count view — widths ∝ rolled-up |Δ GPU activity count| per step; color = signed Δ",
    "IDLE_H2": "Sum + idle attribution — widths ∝ rolled-up |Δ activity or idle time| by enclosing scope; color = signed Δ",
    "V1NAME": "sum diff",
    "V2NAME": "solo diff",
    "DIFFNAME": "net delta",
    "DIFFFORMULA": "(after - before)",
}


def write_html(
    root: StackNode,
    root_idle: StackNode | None,
    out_path: Path,
    title: str,
) -> None:
    """Write the interactive HTML flame graph.

    `root` (pure GPU activity) drives the Sum + Solo charts; `root_idle` (same
    tree with idle added) drives the third "Sum + idle" chart, or `null`/no third
    chart when there is no idle.
    """
    data = json.dumps(_tree_to_json(root), ensure_ascii=False)
    data_idle = (
        json.dumps(_tree_to_json(root_idle), ensure_ascii=False)
        if root_idle is not None
        else "null"
    )
    html = (
        FLAMEGRAPH_HTML_TEMPLATE.replace("__TITLE__", _html_esc(title))
        .replace("__DATA_IDLE__", data_idle)
        .replace("__DATA__", data)
    )
    for key, val in LABELS.items():
        html = html.replace(f"__{key}__", val)
    out_path.write_text(html)


# =============================================================================
# Differential flame graph (compare two stack trees: baseline vs current) —
# four Diff-only views (Solo/Sum/Count/Sum+idle). Widths are absolute leaf
# deltas rolled up through the hierarchy; colors use signed net deltas.

# =============================================================================

def _merge_metrics_json(
    a: "StackNode | None",
    b: "StackNode | None",
    scale_a: float,
    scale_b: float,
) -> dict:
    """Build a compact, server-side precomputed Diff tree.

    ``d`` is the signed after-before delta. Leaf ``v`` values are ``abs(d)``;
    internal ``v`` values sum their children, preserving the additive hierarchy
    required by an ordinary flame graph. Both arrays use (sum, solo, count) order.
    """
    ref = a if a is not None else b
    assert ref is not None
    label = frame_tag(ref.name, ref.kind) if ref.kind != "root" else "all"

    def metrics(node: "StackNode | None", scale: float) -> tuple[float, float, float]:
        if node is None:
            return (0.0, 0.0, 0.0)
        return (node.value * scale, node.solo_value * scale, node.count * scale)

    av = metrics(a, scale_a)
    bv = metrics(b, scale_b)
    delta = tuple(bv[i] - av[i] for i in range(3))
    node = {"n": label, "k": ref.kind, "d": delta}

    keys: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for src in (a, b):
        if src is not None:
            for key in src.children:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
    if keys:
        children = [
            _merge_metrics_json(
                a.children.get(key) if a is not None else None,
                b.children.get(key) if b is not None else None,
                scale_a,
                scale_b,
            )
            for key in keys
        ]
        children.sort(key=lambda child: -child["v"][0])
        node["c"] = children
        node["v"] = tuple(
            sum(child["v"][i] for child in children) for i in range(3)
        )
    else:
        node["v"] = tuple(abs(value) for value in delta)
    return node


def _count_diff_nodes(root: dict) -> int:
    return 1 + sum(_count_diff_nodes(child) for child in root.get("c", ()))


def _prune_diff_tree(root: dict, min_fraction: float) -> tuple[int, int]:
    """Prune insignificant Diff subtrees and aggregate them per parent.

    A child is retained when its rolled-up magnitude reaches ``min_fraction``
    of the root magnitude in at least one of (sum, solo, count). Removed siblings
    become one ``<other diff>`` leaf, preserving both displayed widths and signed
    net deltas exactly.
    """
    before = _count_diff_nodes(root)
    if min_fraction <= 0:
        return before, before

    denominators = tuple(max(float(value), 1.0) for value in root["v"])

    def significant(node: dict) -> bool:
        return any(
            node["v"][i] / denominators[i] >= min_fraction for i in range(3)
        )

    def rec(node: dict) -> None:
        children = node.get("c")
        if not children:
            return
        kept: list[dict] = []
        dropped: list[dict] = []
        for child in children:
            (kept if significant(child) else dropped).append(child)
        for child in kept:
            rec(child)
        if dropped:
            values = tuple(sum(child["v"][i] for child in dropped) for i in range(3))
            if any(values):
                kept.append(
                    {
                        "n": f"<other diff: {len(dropped)} branches>",
                        "k": "other",
                        "v": values,
                        "d": tuple(
                            sum(child["d"][i] for child in dropped)
                            for i in range(3)
                        ),
                    }
                )
        kept.sort(key=lambda child: -child["v"][0])
        node["c"] = kept

    rec(root)
    return before, _count_diff_nodes(root)


def write_diff_html(
    tree_a: StackNode,
    tree_b: StackNode,
    tree_a_idle: StackNode | None,
    tree_b_idle: StackNode | None,
    out_path: Path,
    title: str,
    label_a: str,
    label_b: str,
    scale_a: float,
    scale_b: float,
    min_fraction: float = 0.001,
) -> tuple[int, int, int, int]:
    """Write four Diff-only views (Solo/Sum/Count/Sum+idle).

    Widths are absolute leaf deltas rolled up through the hierarchy; colors encode
    signed net delta (red = grew, blue = shrank). Diff and non-Diff outputs use
    the same HTML renderer; its compact Diff data is projected to the normal
    node schema in the browser.
    """
    main_tree = _merge_metrics_json(tree_a, tree_b, scale_a, scale_b)
    main_before, main_after = _prune_diff_tree(main_tree, min_fraction)
    data = json.dumps(main_tree, ensure_ascii=False, separators=(",", ":"))
    if tree_a_idle is not None and tree_b_idle is not None:
        idle_tree = _merge_metrics_json(
            tree_a_idle, tree_b_idle, scale_a, scale_b
        )
        idle_before, idle_after = _prune_diff_tree(idle_tree, min_fraction)
        data_idle = json.dumps(
            idle_tree, ensure_ascii=False, separators=(",", ":")
        )
    else:
        idle_before = idle_after = 0
        data_idle = "null"
    html = (
        FLAMEGRAPH_HTML_TEMPLATE.replace("__TITLE__", _html_esc(title))
        .replace("__DATA_IDLE__", data_idle)
        .replace("__DATA__", data)
    )
    labels = dict(DIFF_LABELS)
    labels["NOTE"] += (
        "<br>before = <b>" + _html_esc(label_a) + "</b> &nbsp;·&nbsp; "
        "after = <b>" + _html_esc(label_b) + "</b>"
        f"<br>Backend pruning threshold: {100 * min_fraction:g}% of total "
        "change in all views; omitted siblings are preserved as &lt;other diff&gt;."
    )
    for key, val in labels.items():
        html = html.replace(f"__{key}__", val)
    out_path.write_text(html)
    return main_before, main_after, idle_before, idle_after


# =============================================================================
# SQL
# =============================================================================

KERNELS_IN_WINDOW_SQL = (
    KERNEL_SQL
    + """
WHERE k.start >= ? AND k.end <= ?
ORDER BY r.start
"""
)

# GPU memory copies / sets: separate CUPTI activity tables (NOT joined by
# KERNEL_SQL), so they are invisible to a kernel-only tree. Each row joins its
# launch API (cudaMemcpy*/cudaMemset*) via correlationId so the enclosing NVTX
# stack resolves through the same mapping as kernels; copyKind → readable label.
MEMCPY_IN_WINDOW_SQL = """
SELECT
    m.start           AS gpu_start,
    m.end             AS gpu_end,
    m.end - m.start   AS gpu_dur_ns,
    CASE m.copyKind
        WHEN 1  THEN 'Memcpy HtoD'
        WHEN 2  THEN 'Memcpy DtoH'
        WHEN 8  THEN 'Memcpy DtoD'
        WHEN 9  THEN 'Memcpy HtoH'
        WHEN 10 THEN 'Memcpy PtoP'
        ELSE 'Memcpy kind=' || m.copyKind
    END               AS kernel_name,
    r.start           AS api_start,
    r.end             AS api_end
FROM CUPTI_ACTIVITY_KIND_MEMCPY m
JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = m.correlationId
WHERE m.start >= ? AND m.end <= ?
ORDER BY r.start
"""

MEMSET_IN_WINDOW_SQL = """
SELECT
    m.start           AS gpu_start,
    m.end             AS gpu_end,
    m.end - m.start   AS gpu_dur_ns,
    'Memset'          AS kernel_name,
    r.start           AS api_start,
    r.end             AS api_end
FROM CUPTI_ACTIVITY_KIND_MEMSET m
JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = m.correlationId
WHERE m.start >= ? AND m.end <= ?
ORDER BY r.start
"""

TRACE_START_SQL = "SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL"


NVTX_TAIL = re.compile(
    r"\s*,\s*(op_id|seq|sizes|input_op_ids|input_shapes|dtype|count)\b.*$"
)

# Type aliases for the flame-graph stack tree.
Frame = tuple[str, str]  # (display name, kind: "nvtx" | "gpu")
FramePath = list[Frame]  # a root->leaf path of frames (NVTX scopes + GPU leaf)
Interval = tuple[int, int]  # a [start, end) time interval in ns
StepMarker = tuple[int, int, str]  # (start, end, normalized marker name)


def _merge(intervals: list[Interval]) -> list[Interval]:
    """Sort and merge overlapping/adjacent [start, end) intervals."""
    if not intervals:
        return []
    intervals = sorted(intervals)
    out: list[Interval] = [intervals[0]]
    for s, e in intervals[1:]:
        ls, le = out[-1]
        if s > le:
            out.append((s, e))
        elif e > le:
            out[-1] = (ls, e)
    return out


def union_len(intervals: list[Interval]) -> int:
    """Total length of the union of [start, end) intervals."""
    return sum(e - s for s, e in _merge(intervals))


def normalize(name: str) -> str:
    return NVTX_TAIL.sub("", name)


def detect_steps(idx: NvtxIndex, needle: str) -> list[StepMarker]:
    """
    Unique (start, end, normalized_name) step markers, sorted by end time.
    """
    matches = idx.matches(needle)

    seen: set[tuple[int, int]] = set()
    out: list[StepMarker] = []

    for s, e, name in matches:
        key = (s, e)
        if key not in seen:
            seen.add(key)
            out.append((s, e, normalize(name)))

    out.sort(key=lambda t: t[1])
    return out


def compute_step_windows(
    steps: list[StepMarker], t_min: int, skip_steps: int
) -> tuple[list[StepMarker], int, int]:
    """Turn step-end markers into per-step [start, end] windows + the analysis window.

    Step k spans (end of step k-1, end of step k]; step 1 starts at the trace
    origin. Returns (step_windows, window_start, window_end) where the analysis
    window covers everything after the skipped warmup steps.
    """
    step_windows: list[StepMarker] = []
    prev_end = t_min
    for _, e, name in steps:
        step_windows.append((prev_end, e, name))
        prev_end = e
    window_start = step_windows[skip_steps - 1][1] if skip_steps > 0 else t_min
    window_end = step_windows[-1][1]
    return step_windows, window_start, window_end


def print_header_and_steps(
    db_path: str,
    rank: int | None,
    step_needle: str,
    steps: list[StepMarker],
    step_windows: list[StepMarker],
    skip_steps: int,
    t_min: int,
    window_dur: int,
) -> None:
    """Print the profile header and the per-step window table."""
    distinct_markers = sorted({w[2] for w in step_windows})

    print(f"Profile : {db_path}")
    if rank is not None:
        print(f"Rank    : {rank}")
    print(
        f"Steps detected via NVTX containing '{step_needle}': {len(steps)} "
        f"(skipping first {skip_steps})"
    )
    if len(distinct_markers) > 1:
        print(
            f"Note: '{step_needle}' matches {len(distinct_markers)} distinct NVTX names — "
            "step rows below may alternate between them. If you want one row per "
            "training iteration, pass a more specific --step-nvtx, e.g.:"
        )
        for m in distinct_markers:
            print(f"    --step-nvtx {m!r}")
    print()

    marker_w = max(
        MARKER_W_MIN,
        min(MARKER_W_MAX, max((len(w[2]) for w in step_windows), default=MARKER_W_MIN)),
    )
    print(
        f"{'step':>4}  {'window (ms from trace start)':<32}  "
        f"{'wall':>12}  {'marker':<{marker_w}}  note"
    )
    for i, (ws, we, name) in enumerate(step_windows, 1):
        note = "warmup (skipped)" if i <= skip_steps else ""
        print(
            f"{i:>4}  [{(ws - t_min) / NS_PER_MS:>10.2f} -> {(we - t_min) / NS_PER_MS:>10.2f}]  "
            f"{human_ns(we - ws):>12}  {truncate(name, marker_w):<{marker_w}}  {note}"
        )
    print(
        f"\nPost-warmup window: steps {skip_steps + 1}..{len(steps)} "
        f"= {human_ns(window_dur)} wall\n"
    )


def rows_to_activities(rows, kind: str) -> list[tuple]:
    """Turn a KERNEL/MEMCPY/MEMSET result set into flame-tree activity tuples.

    Each tuple is (api_start, api_end, gpu_start, gpu_end, gpu_dur_ns, name, kind);
    the `kind` becomes the leaf frame's kind ("gpu" / "memcpy" / "memset").
    """
    return [
        (
            r["api_start"],
            r["api_end"],
            r["gpu_start"],
            r["gpu_end"],
            r["gpu_dur_ns"],
            r["kernel_name"],
            kind,
        )
        for r in rows
    ]


def _stack_to_nvtx_frames(stack, stack_depth: int) -> FramePath:
    """Enclosing NVTX stack -> flame-tree frames (outermost-first, depth-capped)."""
    if not stack:
        nvtx_frames: FramePath = [("<no nvtx>", "nvtx")]
    else:
        nvtx_frames = [(normalize(e.name), "nvtx") for e in stack]
    if stack_depth > 0:
        nvtx_frames = nvtx_frames[:stack_depth]
    return nvtx_frames


def build_flame_tree(
    idx: NvtxIndex,
    activities: list[tuple],
    stack_depth: int,
) -> tuple[StackNode, list[Interval]]:
    """
    Resolve each in-window GPU activity's enclosing NVTX stack and build the tree.

    `activities` are (api_start, api_end, gpu_start, gpu_end, dur, name, kind)
    tuples — kernels and, when included, memcpy/memset. Solo/idle accounting is
    over the union of *all* activities, so memcpy-only spans no longer read as
    idle.
    """
    items = sorted(activities, key=lambda t: t[0])

    # path, gpu_dur_ns
    leaf_paths: list[tuple[FramePath, int]] = []
    gpu_intervals: list[Interval] = []

    stack_iter = idx.iter_stacks((a, b) for a, b, _, _, _, _, _ in items)
    for (_, _, g_s, g_e, dur, name, kind), (_, _, stack) in zip(items, stack_iter):
        nvtx_frames = _stack_to_nvtx_frames(stack, stack_depth)
        leaf_paths.append((nvtx_frames + [(name, kind)], dur))
        gpu_intervals.append((g_s, g_e))

    solos = compute_solo_times(gpu_intervals)
    tree = StackNode("<root>", "root")
    for (path, dur), solo in zip(leaf_paths, solos):
        tree.add_path(path, dur, solo)
    return tree, gpu_intervals


def _idle_gaps(activity_intervals: list[Interval], ws: int, we: int) -> list[Interval]:
    """Maximal [start, end) spans in [ws, we) covered by no GPU activity."""
    gaps: list[Interval] = []
    prev = ws
    for s, e in _merge(activity_intervals):
        if s > prev:
            gaps.append((prev, s))
        prev = max(prev, e)
    if we > prev:
        gaps.append((prev, we))
    return gaps


def add_attributed_idle(
    idx: NvtxIndex,
    tree: StackNode,
    activity_intervals: list[Interval],
    ws: int,
    we: int,
    stack_depth: int,
) -> int:
    """Distribute GPU-idle gaps under their enclosing NVTX scope as `<idle>` leaves.

    Each gap is attributed to the innermost NVTX frame that fully encloses it
    (resolved via the same sweep-line as kernels, keyed on GPU wall-time — idle
    has no launch site, so attribution is approximate under heavy CPU-ahead
    async). Gaps outside any NVTX frame land under `<no nvtx>`. Returns total
    idle attributed (ns). Idle never overlaps GPU work, so solo == value.
    """
    gaps = _idle_gaps(activity_intervals, ws, we)
    if not gaps:
        return 0
    total = 0
    stack_iter = idx.iter_stacks(iter(gaps))
    for (gs, ge), (_, _, stack) in zip(gaps, stack_iter):
        dur = ge - gs
        if dur <= 0:
            continue
        nvtx_frames = _stack_to_nvtx_frames(stack, stack_depth)
        tree.add_path(nvtx_frames + [("<idle>", "idle")], dur, dur)
        total += dur
    return total


def _resolve_html_path(flamegraph_out: str) -> Path:
    """Turn a user --flamegraph OUT into the concrete '<OUT>.html' path."""
    base = Path(flamegraph_out)
    if base.suffix.lower() in (".html", ".folded"):
        return base.with_suffix(".html")
    return Path(str(base) + ".html")


def write_report_flamegraph(
    tree: StackNode,
    tree_idle: StackNode | None,
    flamegraph_out: str,
    db_path: str,
    rank: int | None,
    skip_steps: int,
    n_steps: int,
) -> None:
    """
    Write the interactive HTML flame graph for the kernel->NVTX stack tree.
    """
    html = _resolve_html_path(flamegraph_out)
    title = (
        f"Kernel stack flame graph — {Path(db_path).name}"
        + (f" (rank {rank})" if rank is not None else "")
        + f" | post-warmup steps {skip_steps + 1}..{n_steps}"
    )
    write_html(tree, tree_idle, html, title)
    print(
        f"Wrote flame graph: {html}   "
        "(open in browser — click-to-zoom, search, tooltip)"
    )
    print()


def load_flame_tree(
    db_path: str,
    step_nvtx: str,
    skip_steps: int,
    stack_depth: int,
    *,
    print_steps: bool,
    include_memcpy: bool = True,
    attribute_idle: bool = True,
) -> tuple[StackNode | None, StackNode | None, int, int | None]:
    """Detect steps and build the post-warmup flame tree(s).

    Returns (tree, tree_idle, n_steps, rank). `tree` is the pure GPU-activity
    tree (kernels + optional memcpy/memset, NO idle) that drives the Sum/Solo
    charts; `tree_idle` is a clone of it with GPU-idle added as `<idle>` leaves
    (attributed under each gap's enclosing NVTX scope when `attribute_idle`, else
    one root block) — it drives the third "Sum + idle" chart. `tree` is None only
    when the window has no GPU activity; `tree_idle` is None when there is no
    idle. When `print_steps` is set the per-step window table is printed.
    """
    conn = open_db(db_path)
    require_kernel_table(conn, db_path)
    rank = get_rank(conn)

    idx = NvtxIndex(conn, rank)
    steps = detect_steps(idx, step_nvtx)
    if len(steps) <= skip_steps:
        conn.close()
        print(
            f"Error: found {len(steps)} NVTX step markers matching '{step_nvtx}' "
            f"in {db_path}, cannot skip {skip_steps} and still have any steps left.",
            file=sys.stderr,
        )
        sys.exit(1)

    t_min = conn.execute(TRACE_START_SQL).fetchone()[0]
    step_windows, window_start, window_end = compute_step_windows(
        steps, t_min, skip_steps
    )
    window_dur = window_end - window_start

    if print_steps:
        print_header_and_steps(
            db_path, rank, step_nvtx, steps, step_windows, skip_steps, t_min, window_dur
        )

    tree: StackNode | None = None
    tree_idle: StackNode | None = None
    win = (window_start, window_end)
    rows = conn.execute(KERNELS_IN_WINDOW_SQL, win).fetchall()
    activities = rows_to_activities(rows, "gpu")
    if include_memcpy:
        if has_table(conn, "CUPTI_ACTIVITY_KIND_MEMCPY"):
            activities += rows_to_activities(
                conn.execute(MEMCPY_IN_WINDOW_SQL, win).fetchall(), "memcpy"
            )
        if has_table(conn, "CUPTI_ACTIVITY_KIND_MEMSET"):
            activities += rows_to_activities(
                conn.execute(MEMSET_IN_WINDOW_SQL, win).fetchall(), "memset"
            )
    if activities:
        tree, gpu_intervals = build_flame_tree(idx, activities, stack_depth)
        idle = max(0, window_dur - union_len(gpu_intervals))
        if idle > 0:
            # idle lives ONLY in the third chart: clone the activity tree and add
            # idle there, leaving the Sum/Solo tree pure GPU activity.
            tree_idle = tree.clone()
            if attribute_idle:
                add_attributed_idle(
                    idx,
                    tree_idle,
                    gpu_intervals,
                    window_start,
                    window_end,
                    stack_depth,
                )
            else:
                tree_idle.add_path([("<idle>", "idle")], idle, idle)
    conn.close()
    return tree, tree_idle, len(steps), rank


def run_single(args: argparse.Namespace) -> None:
    """Standard single-profile flow: step table + optional flame graph."""
    tree, tree_idle, n_steps, rank = load_flame_tree(
        args.db,
        args.step_nvtx,
        args.skip_steps,
        args.stack_depth,
        print_steps=True,
        include_memcpy=args.include_memcpy,
        attribute_idle=args.attribute_idle,
    )
    if not args.flamegraph:
        return
    if tree is None:
        print("No kernels in post-warmup window.")
        return
    write_report_flamegraph(
        tree, tree_idle, args.flamegraph, args.db, rank, args.skip_steps, n_steps
    )


def run_diff(args: argparse.Namespace) -> None:
    """Differential flow: build both trees and emit a diff flame graph.

    `args.db` is the current/"after" profile; `args.diff` is the baseline/
    "before". Values are normalized per post-warmup step so windows spanning
    different step counts compare fairly; the delta `after - before` colors each
    frame (red = grew, blue = shrank).
    """
    if not args.flamegraph:
        print(
            "Error: --diff requires --flamegraph OUT to write the comparison to.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"=== current (after)  : {args.db} ===")
    tree_b0, tree_b_idle, n_steps_b, _ = load_flame_tree(
        args.db,
        args.step_nvtx,
        args.skip_steps,
        args.stack_depth,
        print_steps=True,
        include_memcpy=args.include_memcpy,
        attribute_idle=args.attribute_idle,
    )
    print(f"=== baseline (before): {args.diff} ===")
    tree_a0, tree_a_idle, n_steps_a, _ = load_flame_tree(
        args.diff,
        args.step_nvtx,
        args.skip_steps,
        args.stack_depth,
        print_steps=True,
        include_memcpy=args.include_memcpy,
        attribute_idle=args.attribute_idle,
    )

    # Solo/Sum/Count come from the pure-activity trees; Sum+idle from the
    # idle-attributed trees (present only when both windows had idle).
    if tree_a0 is None or tree_b0 is None:
        print("No kernels in one of the post-warmup windows — nothing to diff.")
        return

    steps_a = max(1, n_steps_a - args.skip_steps)
    steps_b = max(1, n_steps_b - args.skip_steps)
    scale_a = 1.0 / steps_a
    scale_b = 1.0 / steps_b

    html = _resolve_html_path(args.flamegraph)
    label_a = Path(args.diff).name
    label_b = Path(args.db).name
    title = f"Differential flame graph — {label_b} vs {label_a} (per-step, Solo/Sum/Count/Sum+idle)"
    prune_stats = write_diff_html(
        tree_a0,
        tree_b0,
        tree_a_idle,
        tree_b_idle,
        html,
        title,
        label_a,
        label_b,
        scale_a,
        scale_b,
        args.diff_min_percent / 100.0,
    )
    print(
        "Backend Diff pruning: "
        f"main {prune_stats[0]} -> {prune_stats[1]} nodes; "
        f"idle {prune_stats[2]} -> {prune_stats[3]} nodes "
        f"(threshold {args.diff_min_percent:g}%)"
    )
    print(
        f"Baseline steps/window: {steps_a}   Current steps/window: {steps_b}   "
        "(values normalized to per-step averages)"
    )
    print(
        f"Wrote diff flame graph: {html}   "
        "(Diff only: red = grew, blue = shrank; "
        "Solo / Sum / Count / Sum+idle)"
    )
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect training-step boundaries; optionally emit a kernel->NVTX flame graph"
    )
    p.add_argument("db", help="Path to .sqlite profile")
    p.add_argument(
        "--step-nvtx",
        default="Optimizer.step",
        metavar="SUBSTR",
        help="NVTX name substring that marks one training step boundary "
        "(default: 'Optimizer.step')",
    )
    p.add_argument(
        "--skip-steps",
        type=int,
        default=1,
        metavar="N",
        help="Number of warmup steps to skip from the start (default: 1)",
    )
    p.add_argument(
        "--stack-depth",
        type=int,
        default=0,
        metavar="N",
        help="Limit the CPU NVTX stack depth (outermost-first) in the flame graph. "
        "0 = full stack (default). The GPU kernel is always the leaf — NVTX is "
        "CPU semantic markup, the kernel is the real work.",
    )
    p.add_argument(
        "--flamegraph",
        default=None,
        metavar="OUT",
        help="Write an interactive flame graph over the post-warmup window to "
        "'<OUT>.html' (d3-flame-graph, click-to-zoom / search / tooltip; loads d3 "
        "from CDN). CPU NVTX frames are cool-colored; GPU kernel leaves are "
        "warm-colored.",
    )
    p.add_argument(
        "--diff",
        default=None,
        metavar="BASELINE_DB",
        help="Compare against a baseline profile: emit a differential flame graph "
        "to '<OUT>.html' (requires --flamegraph). The positional 'db' is the "
        "current/'after' profile; BASELINE_DB is the 'before'. Values are "
        "normalized per post-warmup step, and each frame is colored by the delta "
        "(red = more GPU time now, blue = less). Widths show rolled-up absolute "
        "change magnitude, so removed and added frames are both visible.",
    )
    p.add_argument(
        "--diff-min-percent",
        type=float,
        default=1.0,
        metavar="PCT",
        help="Backend-prune Diff subtrees smaller than this percentage of total "
        "change in all views, aggregating omitted siblings as <other diff> "
        "(default: 1.0; 0 disables pruning).",
    )
    p.add_argument(
        "--include-memcpy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include GPU memcpy (H2D/D2H/D2D) and memset activities as flame-graph "
        "leaves, not just compute kernels (default: on). These live in separate "
        "CUPTI tables; with them off, memcpy-only spans count as <idle>. Use "
        "--no-include-memcpy to restore compute-only behavior.",
    )
    p.add_argument(
        "--attribute-idle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Distribute GPU-idle gaps under the NVTX scope enclosing each gap "
        "(so you can see which phase the GPU stalled in), rendered as <idle> "
        "leaves throughout the tree (default: on). --no-attribute-idle instead "
        "lumps all idle into one <idle> block at the root.",
    )
    args = p.parse_args()
    if args.skip_steps < 0:
        p.error("--skip-steps must be >= 0")
    if args.stack_depth < 0:
        p.error("--stack-depth must be >= 0")
    if args.diff_min_percent < 0:
        p.error("--diff-min-percent must be >= 0")
    return args


if __name__ == "__main__":
    args = parse_args()

    """
    STEP1. Detect step markers and compute the post-warmup analysis window.
    STEP2. Read GPU kernels whose GPU execution interval falls inside that window.
    STEP3. Build the flame tree:
        - Use each kernel's CUDA launch API interval to look up the enclosing CPU-side NVTX stack.
    STEP4. (--diff) Repeat STEP1-3 for the baseline profile and emit a diff graph.
    """

    if args.diff:
        run_diff(args)
    else:
        run_single(args)
