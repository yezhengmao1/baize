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
    __slots__ = ("name", "kind", "value", "solo_value", "children")

    def __init__(self, name: str, kind: str):
        self.name = name
        self.kind = kind
        self.value = 0
        self.solo_value = 0
        self.children: dict[tuple[str, str], "StackNode"] = {}

    def add_path(self, frames: list[tuple[str, str]], value: int, solo: int) -> None:
        self.value += value
        self.solo_value += solo
        cur = self
        for name, kind in frames:
            key = (name, kind)
            c = cur.children.get(key)
            if c is None:
                c = StackNode(name, kind)
                cur.children[key] = c
            c.value += value
            c.solo_value += solo
            cur = c


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

    Each node carries the two additive quantities (sum / solo); the client-side
    JS picks one as d.value per view, and derives overlap (= sum - solo) for
    the Sum view's color saturation.
    """

    def rec(node: StackNode) -> dict:
        label = frame_tag(node.name, node.kind) if node.kind != "root" else "all"
        out = {
            "name": label,
            "kind": node.kind,
            "sum": node.value,
            "solo": node.solo_value,
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
        '  <span class="sw" style="background:hsl(320,85%,55%)"></span> GPU idle '
        "(no kernel running) — bright magenta so it is unmistakable; gray is "
        "reserved for the <i>overlap</i> encoding in the Sum view\n"
    ),
    "NOTE": (
        "\n  Two charts below share the same hierarchy. Identity per node: "
        "<b>sum = solo + overlap</b>.<br>\n"
        "  In <b>Sum</b>, widths are total kernel time and color saturation "
        "encodes overlap "
        "(vivid = exposed, pale = mostly hidden behind other GPU work).<br>\n"
        "  In <b>Solo</b>, widths are per-kernel solo time — the portion that "
        "actually blocks the GPU clock.<br>\n"
        "  Wide + pale in Sum, narrow in Solo → good concurrency. Wide in both → "
        "fully exposed, optimization target.<br>\n"
        "  Tooltip shows absolute sum / solo / overlap%.\n"
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
    "V1NAME": "sum",
    "V2NAME": "solo",
    "DIFFNAME": "overlap ratio",
    "DIFFFORMULA": "(1 - solo/sum)",
}


def write_html(root: StackNode, out_path: Path, title: str) -> None:
    """Write the interactive HTML flame graph for the kernel->NVTX stack tree."""
    data = json.dumps(_tree_to_json(root), ensure_ascii=False)
    html = FLAMEGRAPH_HTML_TEMPLATE.replace("__TITLE__", _html_esc(title)).replace(
        "__DATA__", data
    )
    for key, val in LABELS.items():
        html = html.replace(f"__{key}__", val)
    out_path.write_text(html)


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


def build_flame_tree(
    idx: NvtxIndex,
    rows,
    stack_depth: int,
) -> tuple[StackNode, list[Interval]]:
    """
    Resolve each in-window kernel's enclosing NVTX stack and build the flame tree.
    """
    items = [
        (
            r["api_start"],
            r["api_end"],
            r["gpu_start"],
            r["gpu_end"],
            r["gpu_dur_ns"],
            r["kernel_name"],
        )
        for r in rows
    ]
    items.sort(key=lambda t: t[0])

    # path, gpu_dur_ns
    kernel_paths: list[tuple[FramePath, int]] = []
    gpu_intervals: list[Interval] = []

    stack_iter = idx.iter_stacks((a, b) for a, b, _, _, _, _ in items)
    for (_, _, g_s, g_e, k_dur, k_name), (_, _, stack) in zip(items, stack_iter):
        if not stack:
            nvtx_frames: FramePath = [("<no nvtx>", "nvtx")]
        else:
            nvtx_frames = [(normalize(e.name), "nvtx") for e in stack]

        if stack_depth > 0:
            nvtx_frames = nvtx_frames[:stack_depth]
        kernel_paths.append((nvtx_frames + [(k_name, "gpu")], k_dur))
        gpu_intervals.append((g_s, g_e))

    solos = compute_solo_times(gpu_intervals)
    tree = StackNode("<root>", "root")
    for (path, k_dur), solo in zip(kernel_paths, solos):
        tree.add_path(path, k_dur, solo)
    return tree, gpu_intervals


def write_report_flamegraph(
    tree: StackNode,
    idle: int,
    flamegraph_out: str,
    db_path: str,
    rank: int | None,
    skip_steps: int,
    n_steps: int,
) -> None:
    """
    Write the interactive HTML flame graph for the kernel->NVTX stack tree.
    """
    if idle > 0:
        tree.add_path([("<idle>", "idle")], idle, idle)
    base = Path(flamegraph_out)
    if base.suffix.lower() in (".html", ".folded"):
        html = base.with_suffix(".html")
    else:
        html = Path(str(base) + ".html")
    title = (
        f"Kernel stack flame graph — {Path(db_path).name}"
        + (f" (rank {rank})" if rank is not None else "")
        + f" | post-warmup steps {skip_steps + 1}..{n_steps}"
    )
    write_html(tree, html, title)
    print(
        f"Wrote flame graph: {html}   "
        "(open in browser — click-to-zoom, search, tooltip)"
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
    args = p.parse_args()
    if args.skip_steps < 0:
        p.error("--skip-steps must be >= 0")
    if args.stack_depth < 0:
        p.error("--stack-depth must be >= 0")
    return args


if __name__ == "__main__":
    args = parse_args()

    conn = open_db(args.db)
    require_kernel_table(conn, args.db)
    rank = get_rank(conn)

    """
    STEP1. Detect step markers and compute the post-warmup analysis window.
    STEP2. Read GPU kernels whose GPU execution interval falls inside that window.
    STEP3. Build the flame tree:
        - Use each kernel's CUDA launch API interval to look up the enclosing CPU-side NVTX stack.
    """

    idx = NvtxIndex(conn, rank)
    steps = detect_steps(idx, args.step_nvtx)
    if len(steps) <= args.skip_steps:
        conn.close()
        print(
            f"Error: found {len(steps)} NVTX step markers matching '{args.step_nvtx}', "
            f"cannot skip {args.skip_steps} and still have any steps left.",
            file=sys.stderr,
        )
        sys.exit(1)

    t_min = conn.execute(TRACE_START_SQL).fetchone()[0]
    step_windows, window_start, window_end = compute_step_windows(
        steps,
        t_min,
        args.skip_steps,
    )
    window_dur = window_end - window_start

    # the step windows is a list of tuples, each with (start, end, name)
    print_header_and_steps(
        args.db,
        rank,
        args.step_nvtx,
        steps,
        step_windows,
        args.skip_steps,
        t_min,
        window_dur,
    )

    if args.flamegraph:
        rows = conn.execute(
            KERNELS_IN_WINDOW_SQL,
            (window_start, window_end),
        ).fetchall()

        if not rows:
            print("No kernels in post-warmup window.")
        else:
            tree, gpu_intervals = build_flame_tree(idx, rows, args.stack_depth)
            idle = max(0, window_dur - union_len(gpu_intervals))
            write_report_flamegraph(
                tree,
                idle,
                args.flamegraph,
                args.db,
                rank,
                args.skip_steps,
                len(steps),
            )

    conn.close()
