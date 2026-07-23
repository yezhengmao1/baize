"""
gpu-shape: per-operator input-shape table (compute only; communication excluded).

For every GPU kernel in the post-warmup window, resolve its innermost enclosing
aten-operator NVTX range — the frame carrying a ``sizes = [[...]]`` shape
annotation, emitted by ``torch.autograd.profiler.emit_nvtx(record_shapes=True)``
— and aggregate kernel GPU time by (operator, input-shapes). Communication
kernels / scopes (nccl / c10d / record_param_comms / deep_ep / Fused{Dispatch,
Combine}) are excluded, so the table is pure compute.

Kernel-first, NVTX-via-the-mapping: kernels are read from CUPTI and their op
context is looked up through the kernel->NVTX stack (utils.nvtx.NvtxIndex); we
never treat NVTX_EVENTS as a first-class table.

Author: yezhengmaolove@gmail.com
"""

import argparse
import csv as csvmod
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
from ..utils.nvtx import NvtxIndex
from .flamegraph import (
    KERNELS_IN_WINDOW_SQL,
    TRACE_START_SQL,
    compute_step_windows,
    detect_steps,
    normalize,
)

NS_PER_MS = 1e6

# Frames whose op name starts with one of these are communication plumbing, not a
# compute operator — skip them when picking the enclosing op frame so the table
# stays compute-only. (Same spirit as utils.comm._PLUMBING, plus the DeepEP MoE
# all-to-all range names, which do carry a `sizes =` field of their own.)
_COMM_SCOPE_PREFIX = (
    "nccl",
    "c10d::",
    "record_param_comms",
    "NcclGroup",
    "deep_ep",
    "FusedDispatch",
    "FusedCombine",
)

# Comm kernels proper (belt-and-suspenders in case one is wrapped in a compute
# scope that happens to carry shapes).
_COMM_KERNEL_PREFIX = ("ncclDevKernel", "ncclKernel")

_WS = re.compile(r"\s+")


def extract_sizes(name: str) -> str | None:
    """Bracket-balanced value of the first ``sizes = `` (or ``input_shapes = ``)
    field in an NVTX op-range name, whitespace-collapsed. None if absent.

    e.g. ``aten::addmm, seq = 5, sizes = [[4096, 4096], [4096, 4096]]``
         -> ``[[4096,4096],[4096,4096]]``
    """
    for key in ("sizes = ", "input_shapes = "):
        i = name.find(key)
        if i < 0:
            continue
        j = i + len(key)
        if j >= len(name) or name[j] != "[":
            continue
        depth = 0
        for k in range(j, len(name)):
            c = name[k]
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    return name[j : k + 1].replace(" ", "")
        return name[j:].replace(" ", "")  # unbalanced; keep the remainder
    return None


def _stack_has_comm(stack) -> bool:
    """True if any enclosing NVTX frame is communication plumbing (NCCL / c10d /
    DeepEP FusedDispatch/Combine …) — used to tag comm kernels that aren't named
    ncclDevKernel (e.g. DeepEP's `dispatch`/`combine`)."""
    return any(normalize(e.name).startswith(_COMM_SCOPE_PREFIX) for e in stack)


def op_frame(stack):
    """Innermost non-comm NVTX frame carrying a shape annotation.

    ``stack`` is outermost-first (as returned by NvtxIndex.iter_stacks), so the
    last qualifying frame is the innermost — the actual op that launched the
    kernel (e.g. aten::addmm inside aten::linear). Returns
    ``(op_name, shapes, start, end)`` or None.
    """
    chosen = None
    for e in stack:
        op = normalize(e.name)
        if op.startswith(_COMM_SCOPE_PREFIX):
            continue
        sizes = extract_sizes(e.name)
        if sizes is None:
            continue
        chosen = (op, sizes, e.start, e.end)
    return chosen


class OpAgg:
    """Aggregate for one (operator, input-shapes) row."""

    __slots__ = ("op", "shapes", "calls", "kernels", "gpu")

    def __init__(self, op: str, shapes: str):
        self.op = op
        self.shapes = shapes
        self.calls = 0
        self.kernels = 0
        self.gpu = 0


def build_shape_table(idx: NvtxIndex, rows):
    """Attribute each in-window compute kernel to its innermost aten-op frame and
    aggregate by (operator, input-shapes).

    Returns (table, stats, kern_names):
      - table: list of OpAgg per (operator, input-shapes).
      - stats: coverage counters (comm / no-shape kernels excluded, and time).
      - kern_names: {shortName: [calls, gpu, mn, mx]} over the **compute** kernels
        only (all communication — NCCL and DeepEP — excluded), for the by-kernel
        view; includes the compute kernels the op table drops as no-shape.
    """
    items = [
        (r["api_start"], r["api_end"], r["gpu_dur_ns"], r["kernel_name"]) for r in rows
    ]
    items.sort(key=lambda t: t[0])

    # One op invocation == one op-frame instance, identified by (op, shapes,
    # start, end). Accumulate kernel time per instance, then roll instances up
    # into per-(op, shapes) rows so `calls` counts invocations, not kernels.
    instances: dict[tuple, list] = {}  # key -> [op, shapes, gpu_sum, n_kernels]
    kern_names: dict[str, list] = {}  # shortName -> [calls, gpu, mn, mx]
    stats = {
        "comm_kernels": 0,
        "comm_gpu": 0,
        "noshape_kernels": 0,
        "noshape_gpu": 0,
        "op_kernels": 0,
        "op_gpu": 0,
    }

    stacks = idx.iter_stacks((a, b) for a, b, _, _ in items)
    for (_, _, k_dur, k_name), (_, _, stack) in zip(items, stacks):
        frame = op_frame(stack)
        # A kernel is communication if it is an NCCL device kernel, or it carries
        # no compute op frame yet sits under comm plumbing (DeepEP dispatch/combine
        # etc.). Comm is excluded from both the op table and the by-kernel view.
        is_comm = k_name.startswith(_COMM_KERNEL_PREFIX) or (
            frame is None and _stack_has_comm(stack)
        )
        if not is_comm:
            kn = kern_names.get(k_name)
            if kn is None:
                kern_names[k_name] = [1, k_dur, k_dur, k_dur]
            else:
                kn[0] += 1
                kn[1] += k_dur
                kn[2] = min(kn[2], k_dur)
                kn[3] = max(kn[3], k_dur)

        if k_name.startswith(_COMM_KERNEL_PREFIX):
            stats["comm_kernels"] += 1
            stats["comm_gpu"] += k_dur
            continue
        if frame is None:
            stats["noshape_kernels"] += 1
            stats["noshape_gpu"] += k_dur
            continue
        op, shapes, s, e = frame
        stats["op_kernels"] += 1
        stats["op_gpu"] += k_dur
        inst = instances.get((op, shapes, s, e))
        if inst is None:
            inst = [op, shapes, 0, 0]
            instances[(op, shapes, s, e)] = inst
        inst[2] += k_dur
        inst[3] += 1

    table: dict[tuple[str, str], OpAgg] = {}
    for op, shapes, gpu_sum, n_kernels in instances.values():
        agg = table.get((op, shapes))
        if agg is None:
            agg = OpAgg(op, shapes)
            table[(op, shapes)] = agg
        agg.calls += 1
        agg.kernels += n_kernels
        agg.gpu += gpu_sum

    return list(table.values()), stats, kern_names


_SORT_KEYS = {
    "time": lambda a: -a.gpu,
    "calls": lambda a: -a.calls,
    "op": lambda a: (a.op, a.shapes),
}


def print_table(
    db_path: str,
    rank: int | None,
    table: list[OpAgg],
    stats: dict,
    n_steps: int,
    skip_steps: int,
    sort: str,
    top: int,
) -> None:
    total_gpu = stats["op_gpu"] or 1
    rows = sorted(table, key=_SORT_KEYS[sort])
    shown = rows[:top] if top > 0 else rows

    print(f"Profile : {db_path}")
    if rank is not None:
        print(f"Rank    : {rank}")
    print(
        f"Window  : post-warmup steps {skip_steps + 1}..{n_steps} "
        f"(compute only; communication excluded)"
    )
    print(
        f"Ops     : {len(table)} distinct (op, shapes); "
        f"{stats['op_kernels']} compute kernels, Σ {human_ns(stats['op_gpu'])} GPU"
    )
    skipped = []
    if stats["comm_kernels"]:
        skipped.append(
            f"{stats['comm_kernels']} comm kernels ({human_ns(stats['comm_gpu'])})"
        )
    if stats["noshape_kernels"]:
        skipped.append(
            f"{stats['noshape_kernels']} no-shape kernels "
            f"({human_ns(stats['noshape_gpu'])})"
        )
    if skipped:
        print(f"Excluded: {'; '.join(skipped)}")
    if not table:
        print(
            "\nNo shape-annotated operators found. Profile with "
            "torch.autograd.profiler.emit_nvtx(record_shapes=True) so aten ops "
            "carry a `sizes = [[...]]` field."
        )
        return
    if top > 0 and len(rows) > top:
        print(f"(showing top {top} of {len(rows)} by {sort})")
    print()

    op_w, sh_w = 26, 46
    print(
        f"{'operator':<{op_w}}  {'input_shapes':<{sh_w}}  "
        f"{'calls':>6}  {'kern':>5}  {'Σ gpu':>11}  {'avg/call':>10}  {'%':>6}"
    )
    print("-" * (op_w + sh_w + 6 + 5 + 11 + 10 + 6 + 12))
    for a in shown:
        pct = 100.0 * a.gpu / total_gpu
        avg = a.gpu / a.calls if a.calls else 0
        print(
            f"{truncate(a.op, op_w):<{op_w}}  {truncate(a.shapes, sh_w):<{sh_w}}  "
            f"{a.calls:>6}  {a.kernels:>5}  {human_ns(a.gpu):>11}  "
            f"{human_ns(avg):>10}  {pct:>5.1f}%"
        )
    print()


def write_csv(table: list[OpAgg], stats: dict, out: str) -> None:
    total_gpu = stats["op_gpu"] or 1
    rows = sorted(table, key=_SORT_KEYS["time"])
    path = Path(out)
    if path.suffix.lower() != ".csv":
        path = Path(str(path) + ".csv")
    with path.open("w", newline="") as f:
        w = csvmod.writer(f)
        w.writerow(
            ["operator", "input_shapes", "calls", "kernels", "gpu_ns", "avg_ns", "pct"]
        )
        for a in rows:
            avg = a.gpu / a.calls if a.calls else 0
            w.writerow(
                [
                    a.op,
                    a.shapes,
                    a.calls,
                    a.kernels,
                    a.gpu,
                    f"{avg:.0f}",
                    f"{100.0 * a.gpu / total_gpu:.3f}",
                ]
            )
    print(f"Wrote shape table CSV: {path} ({len(rows)} rows)")


# =============================================================================
# HTML visualization (ranked bars by operator + sortable table)
# =============================================================================

_SHAPE_TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent / "templates" / "shape_table.html"
)


def _html_esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_shape_html(
    db_path: str,
    rank: int | None,
    table: list[OpAgg],
    stats: dict,
    kernels: dict,
    n_steps: int,
    skip_steps: int,
    out: str,
) -> None:
    """Render the per-(operator, shapes) table as an interactive HTML page.

    `kernels` is the per-kernel-name aggregation `{shortName: [calls, gpu, mn,
    mx]}` from build_shape_table (compute only, all communication excluded) that
    drives the page's "by kernel" view.
    """
    total_gpu = stats["op_gpu"] or 1
    rows = sorted(table, key=_SORT_KEYS["time"])
    rows_json = [
        {
            "op": a.op,
            "shapes": a.shapes,
            "calls": a.calls,
            "kernels": a.kernels,
            "gpu": a.gpu,
            "pct": round(100.0 * a.gpu / total_gpu, 4),
        }
        for a in rows
    ]

    # kernels: {shortName: [calls, gpu, mn, mx]} (compute only, comm excluded)
    total_kern_gpu = sum(v[1] for v in kernels.values()) or 1
    total_kern_calls = sum(v[0] for v in kernels.values())
    kernels_json = [
        {
            "op": name,
            "shapes": None,
            "calls": calls,
            "kernels": calls,
            "gpu": gpu,
            "mn": mn,
            "mx": mx,
            "pct": round(100.0 * gpu / total_kern_gpu, 4),
        }
        for name, (calls, gpu, mn, mx) in sorted(
            kernels.items(), key=lambda kv: -kv[1][1]
        )
    ]

    stats_json = {
        "op_gpu": stats["op_gpu"],
        "op_kernels": stats["op_kernels"],
        "n_ops": len(table),
        "comm_gpu": stats["comm_gpu"],
        "comm_kernels": stats["comm_kernels"],
        "noshape_gpu": stats["noshape_gpu"],
        "noshape_kernels": stats["noshape_kernels"],
        "kern_gpu": total_kern_gpu,
        "kern_names": len(kernels),
        "kern_calls": total_kern_calls,
    }
    title = f"Per-operator input-shape breakdown — {Path(db_path).name}"
    subtitle = (
        (f"rank {rank} · " if rank is not None else "")
        + f"post-warmup steps {skip_steps + 1}..{n_steps} · compute only "
        "(communication excluded) in every view"
    )

    html = (
        _SHAPE_TEMPLATE_PATH.read_text()
        .replace("__TITLE__", _html_esc(title))
        .replace("__SUBTITLE__", _html_esc(subtitle))
        .replace("__STATS__", json.dumps(stats_json))
        .replace("__KERNELS__", json.dumps(kernels_json, ensure_ascii=False))
        .replace("__ROWS__", json.dumps(rows_json, ensure_ascii=False))
    )
    path = Path(out)
    path = (
        path.with_suffix(".html")
        if path.suffix.lower() == ".html"
        else Path(str(path) + ".html")
    )
    path.write_text(html)
    print(f"Wrote shape visualization: {path}   (open in browser)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-operator input-shape table over the post-warmup window "
        "(compute only; communication excluded)."
    )
    p.add_argument("db", help="Path to .sqlite profile")
    p.add_argument(
        "--step-nvtx",
        default="Optimizer.step",
        metavar="SUBSTR",
        help="NVTX name substring marking one training-step boundary "
        "(default: 'Optimizer.step')",
    )
    p.add_argument(
        "--skip-steps",
        type=int,
        default=1,
        metavar="N",
        help="Warmup steps to skip from the start (default: 1)",
    )
    p.add_argument(
        "--sort",
        choices=("time", "calls", "op"),
        default="time",
        help="Row order (default: time = Σ GPU time desc)",
    )
    p.add_argument(
        "--top",
        type=int,
        default=0,
        metavar="N",
        help="Show only the top N rows (0 = all, default)",
    )
    p.add_argument(
        "--csv",
        default=None,
        metavar="OUT",
        help="Write the full (untruncated) table to '<OUT>.csv'",
    )
    p.add_argument(
        "--html",
        default=None,
        metavar="OUT",
        help="Write an interactive HTML visualization to '<OUT>.html' — ranked "
        "bars (top time sinks, by (op,shapes) or aggregated by operator) plus a "
        "sortable/filterable full table; self-contained, light+dark.",
    )
    args = p.parse_args()
    if args.skip_steps < 0:
        p.error("--skip-steps must be >= 0")
    if args.top < 0:
        p.error("--top must be >= 0")
    return args


if __name__ == "__main__":
    args = parse_args()

    conn = open_db(args.db)
    require_kernel_table(conn, args.db)
    rank = get_rank(conn)

    idx = NvtxIndex(conn, rank)
    steps = detect_steps(idx, args.step_nvtx)
    if len(steps) <= args.skip_steps:
        conn.close()
        print(
            f"Error: found {len(steps)} NVTX step markers matching "
            f"'{args.step_nvtx}', cannot skip {args.skip_steps} and still have "
            "any steps left.",
            file=sys.stderr,
        )
        sys.exit(1)

    t_min = conn.execute(TRACE_START_SQL).fetchone()[0]
    _, window_start, window_end = compute_step_windows(steps, t_min, args.skip_steps)

    rows = conn.execute(KERNELS_IN_WINDOW_SQL, (window_start, window_end)).fetchall()
    table, stats, kern_names = build_shape_table(idx, rows)

    print_table(
        args.db,
        rank,
        table,
        stats,
        len(steps),
        args.skip_steps,
        args.sort,
        args.top,
    )
    if args.csv:
        write_csv(table, stats, args.csv)
    if args.html:
        write_shape_html(
            args.db,
            rank,
            table,
            stats,
            kern_names,
            len(steps),
            args.skip_steps,
            args.html,
        )

    conn.close()
