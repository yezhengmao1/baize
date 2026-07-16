"""gpu-deepep-skew: per-EP-group DeepEP and expert-MLP time & skew.

DeepEP-only, cross-rank. Given an EP degree (``--ep``) and a set of per-rank
profiles (the world), the ranks are partitioned into EP groups of ``--ep``
consecutive ranks (``rank // ep`` — the group an all-to-all actually spans; correct
when EP is the fastest-varying parallel axis, i.e. tp=1, which is how DeepEP is
deployed). For each EP group we report, per phase (Dispatch / Combine) and their
Total, the GPU time each rank spent in the A2A over the post-warmup window and the
group's skew:

  floor  = min rank time   — the straggler-free A2A time (a perfectly balanced
                             group would finish in this)
  mean   = average rank time
  max    = slowest rank     — the group is gated on this rank
  skew   = max - floor       — time the fast ranks sit idle waiting on the straggler
  skew%  = skew / max        — fraction of the slow rank's A2A that is imbalance
  strag  = the slowest rank  — who the others wait for

Only the data-moving dispatch/combine kernels carry token volume (from the enclosing
FusedDispatch/FusedCombine NVTX range); the notify/layout helpers move 0 bytes but
their GPU time is still real A2A wait and is counted. Kernels are the primary entity;
volume/scope come through the kernel->NVTX mapping (utils.comm), never a raw
NVTX_EVENTS query.

The MLP section groups kernels by their exact enclosing ``mlp forward``,
``mlp backward``, or ``mlp wgrad`` NVTX instance, measures the earliest-to-latest
GPU kernel span, and aligns instances across ranks by occurrence within the phase.

Author: yezhengmaolove@gmail.com
"""

import argparse
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .flamegraph import compute_step_windows, detect_steps
from ..utils.comm import CommEvent, load_deepep_in_window
from ..utils.common import get_rank, human_ns, open_db, require_kernel_table
from ..utils.kernel import KERNEL_SQL
from ..utils.nvtx import NvtxIndex

LOCAL_WORLD = 8
PHASES = ("Dispatch", "Combine")
MLP_PHASES = ("Forward", "Dgrad", "Wgrad")
MLP_NVTX_TO_PHASE = {
    "mlp forward": "Forward",
    "mlp backward": "Dgrad",
    "mlp wgrad": "Wgrad",
}

TRACE_START_SQL = "SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL"
TRACE_END_SQL = "SELECT MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL"

MLP_KERNEL_SQL = (
    KERNEL_SQL
    + """
WHERE k.start >= ? AND k.start <= ?
ORDER BY r.start
"""
)


@dataclass(frozen=True)
class MlpInstance:
    """One exact MLP NVTX occurrence and the span of kernels launched under it."""

    nvtx_start: int
    nvtx_end: int
    gpu_start: int | None
    gpu_end: int | None
    kernels: int

    @property
    def span(self) -> int | None:
        if self.gpu_start is None or self.gpu_end is None:
            return None
        return self.gpu_end - self.gpu_start


@dataclass(frozen=True)
class MlpPhaseSummary:
    occurrences: int
    floor: float
    mean: float
    maximum: float
    skew: float
    skew_pct: float
    worst_occurrence: int
    worst_skew: int
    worst_skew_pct: float
    worst_straggler: int
    per_rank_mean: dict[int, float]


def human_bytes(b: float) -> str:
    for unit, scale in (
        ("TB", 1 << 40),
        ("GB", 1 << 30),
        ("MB", 1 << 20),
        ("KB", 1 << 10),
    ):
        if b >= scale:
            return f"{b / scale:.2f} {unit}"
    return f"{b:.0f} B"


def human_bw(bytes_: float, ns: float) -> str:
    """Achieved bandwidth bytes/ns -> GB/s (0 if no time / no volume)."""
    if ns <= 0 or bytes_ <= 0:
        return "-"
    return f"{bytes_ / ns:.1f} GB/s"  # bytes/ns == GB/s


# =============================================================================
# Per-rank load (post-warmup window, DeepEP only)
# =============================================================================


def _window(conn, idx: NvtxIndex, needle: str, skip: int) -> tuple[int, int, int]:
    steps = detect_steps(idx, needle)
    t_min = conn.execute(TRACE_START_SQL).fetchone()[0]
    if len(steps) <= skip:
        t_max = conn.execute(TRACE_END_SQL).fetchone()[0]
        return t_min, t_max, len(steps)
    _, window_start, window_end = compute_step_windows(steps, t_min, skip)
    return window_start, window_end, len(steps)


def load_mlp_instances(
    conn, idx: NvtxIndex, ws: int, we: int
) -> dict[str, list[MlpInstance]]:
    """Collect exact MLP NVTX occurrences and their earliest-to-latest GPU span."""
    scratch: dict[tuple[str, int, int], list[int | None]] = {}
    for start, end, name in zip(idx.starts, idx.ends, idx.names):
        phase = MLP_NVTX_TO_PHASE.get(name)
        start, end = int(start), int(end)
        if phase is not None and ws <= start <= we:
            scratch[(phase, start, end)] = [None, None, 0]

    rows = conn.execute(MLP_KERNEL_SQL, (ws, we)).fetchall()
    stacks = idx.iter_stacks((row["api_start"], row["api_end"]) for row in rows)
    for row, (_, _, stack) in zip(rows, stacks):
        target = None
        for frame in reversed(stack):
            phase = MLP_NVTX_TO_PHASE.get(frame.name)
            if phase is not None:
                target = (phase, frame.start, frame.end)
                break
        if target not in scratch:
            continue

        # NOTE: Wgrad may synchronously launch NCCL work that executes on another
        # CUDA stream. For now ownership is defined only by the enclosing NVTX:
        # keep every kernel under "mlp wgrad" and do not classify/filter streams.
        slot = scratch[target]
        gpu_start, gpu_end = row["gpu_start"], row["gpu_end"]
        slot[0] = gpu_start if slot[0] is None else min(slot[0], gpu_start)
        slot[1] = gpu_end if slot[1] is None else max(slot[1], gpu_end)
        slot[2] += 1

    result = {phase: [] for phase in MLP_PHASES}
    for (phase, start, end), (gpu_start, gpu_end, kernels) in scratch.items():
        result[phase].append(MlpInstance(start, end, gpu_start, gpu_end, kernels))
    for instances in result.values():
        instances.sort(key=lambda item: item.nvtx_start)
    return result


def load_rank(
    db: str, needle: str, skip: int, dtype_bytes: int
) -> tuple[int | None, list[CommEvent], int, int]:
    """Load DeepEP dispatch/combine events for one rank in the post-warmup window.
    Returns (rank, events, window_duration_ns, n_steps)."""
    conn = open_db(db)
    require_kernel_table(conn, db)
    rank = get_rank(conn)
    idx = NvtxIndex(conn, rank)
    ws, we, nsteps = _window(conn, idx, needle, skip)
    events = load_deepep_in_window(conn, idx, ws, we, dtype_bytes)
    conn.close()
    events = [
        event._replace(gpu_start=event.gpu_start - ws) for event in events
    ]
    return rank, events, we - ws, nsteps


def load_all(
    dbs: list[str], needle: str, skip: int, dtype_bytes: int, jobs: int
) -> tuple[list[tuple[int, list[CommEvent]]], list[int], int]:
    """Load every rank (ranks are independent files -> optional process pool)."""
    records: list[tuple[int, list[CommEvent]]] = []
    durs: list[int] = []
    nsteps = 0

    def consume(rank, events, wdur, nstep):
        nonlocal nsteps
        if rank is None:
            print(f"  skipped a profile with no RANK env", file=sys.stderr)
            return
        records.append((rank, events))
        durs.append(wdur)
        nsteps = max(nsteps, nstep)
        n_dep = sum(1 for e in events if e.op in PHASES)
        print(f"  loaded rank {rank}: {n_dep} DeepEP kernels", file=sys.stderr)

    if jobs <= 1 or len(dbs) == 1:
        for db in dbs:
            consume(*load_rank(db, needle, skip, dtype_bytes))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(load_rank, db, needle, skip, dtype_bytes) for db in dbs]
            for fut in as_completed(futs):
                consume(*fut.result())
    return records, durs, nsteps


# =============================================================================
# Accumulate -> per rank {phase: [bytes, gpu_dur_ns, calls]}
# =============================================================================


def accumulate(records: list[tuple[int, list[CommEvent]]]) -> dict[int, dict]:
    """Fold DeepEP kernels into per-rank totals and logical Fused* call counts."""
    per_rank: dict[int, dict] = defaultdict(
        lambda: {p: [0, 0, 0] for p in PHASES}  # [bytes, gpu_dur_ns, calls]
    )
    seen_calls: dict[tuple[int, str], set[tuple[int, int]]] = defaultdict(set)
    for rank, events in records:
        for event in events:
            if event.op not in PHASES:
                continue
            slot = per_rank[rank][event.op]
            slot[0] += event.bytes
            slot[1] += event.gpu_dur
            if event.call_id is not None:
                seen_calls[(rank, event.op)].add(event.call_id)
    for rank, phases in per_rank.items():
        for phase in PHASES:
            phases[phase][2] = len(seen_calls[(rank, phase)])
    return per_rank




@dataclass
class LogicalCall:
    """One rank's kernels folded into one enclosing FusedDispatch/Combine call."""

    phase: str
    start_ns: int
    end_ns: int
    gpu_dur_ns: int = 0
    bytes: int = 0


def _logical_calls_by_rank(
    records: list[tuple[int, list[CommEvent]]],
) -> dict[int, list[LogicalCall]]:
    """Fold kernels by their rank-local enclosing Fused* time range."""
    result = {}
    for rank, events in records:
        folded: dict[tuple[str, tuple[int, int]], LogicalCall] = {}
        for event in events:
            if event.op not in PHASES or event.call_id is None:
                continue
            local_key = (event.op, event.call_id)
            event_end = event.gpu_start + event.gpu_dur
            call = folded.get(local_key)
            if call is None:
                call = folded[local_key] = LogicalCall(
                    phase=event.op,
                    start_ns=event.gpu_start,
                    end_ns=event_end,
                )
            call.start_ns = min(call.start_ns, event.gpu_start)
            call.end_ns = max(call.end_ns, event_end)
            call.gpu_dur_ns += event.gpu_dur
            call.bytes += event.bytes
        result[rank] = sorted(folded.values(), key=lambda call: call.start_ns)
    return result


def _match_phase_by_time(
    calls: dict[int, list[LogicalCall]],
    ranks: list[int],
    phase: str,
) -> tuple[list[list[LogicalCall]], int]:

    """Match calls by chronological position within the analysis window.
    This deliberately does not use an NVTX sequence field. Each rank's calls
    are ordered by GPU start time inside the same post-warmup analysis window;
    calls at one chronological position form that position's cross-rank time
    window.
    """
    per_rank = {
        rank: [call for call in calls[rank] if call.phase == phase] for rank in ranks
    }
    counts = {rank: len(per_rank[rank]) for rank in ranks}
    if len(set(counts.values())) != 1:
        raise ValueError(f"{phase}: unequal per-rank logical-call counts {counts}")
    matched = [list(group) for group in zip(*(per_rank[rank] for rank in ranks))]
    return matched, 0


def build_histogram_samples(
    records: list[tuple[int, list[CommEvent]]], ep: int
) -> tuple[dict[str, dict[str, list[float]]], list[str]]:
    """Build per-call samples by chronological GPU time within each analysis window."""
    calls = _logical_calls_by_rank(records)
    groups: dict[int, list[int]] = defaultdict(list)
    for rank in calls:
        groups[rank // ep].append(rank)
    samples = {
        phase: {"total_ns": [], "comm_ns": [], "skew_ns": []} for phase in PHASES
    }
    warnings = []
    for gid, ranks in sorted(groups.items()):
        ranks.sort()
        if len(ranks) != ep:
            warnings.append(
                f"g{gid}: skipped incomplete EP group ({len(ranks)}/{ep} ranks)"
            )
            continue
        group_matches = {}
        for phase in PHASES:
            matched, unmatched = _match_phase_by_time(calls, ranks, phase)
            if unmatched:
                raise ValueError(
                    f"g{gid} {phase}: {unmatched} chronological calls unmatched"
                )
            group_matches[phase] = matched
        phase_counts = {phase: len(group_matches[phase]) for phase in PHASES}
        if len(set(phase_counts.values())) != 1:
            raise ValueError(
                f"g{gid}: Dispatch/Combine logical-call counts differ: {phase_counts}"
            )
        for phase in PHASES:
            for rank_calls in group_matches[phase]:
                times = [call.gpu_dur_ns for call in rank_calls]
                floor, maximum = min(times), max(times)
                samples[phase]["total_ns"].append(float(maximum))
                samples[phase]["comm_ns"].append(float(floor))
                samples[phase]["skew_ns"].append(float(maximum - floor))
    return samples, warnings


def render_histograms(
    records: list[tuple[int, list[CommEvent]]],
    ep: int,
    out: str,
    bins: int | None = None,
) -> tuple[str, list[str]]:
    """Write a two-row publication-style distribution and decomposition figure."""
    if bins is not None and bins < 1:
        raise ValueError("histogram bins must be positive")
    samples, warnings = build_histogram_samples(records, ep)
    if not any(samples[phase]["total_ns"] for phase in PHASES):
        raise ValueError("no complete, cross-rank DeepEP calls available for plotting")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MultipleLocator

    cell_ms = 0.25
    time_max_ms = max(
        max(samples[phase]["total_ns"], default=0) / 1e6 for phase in PHASES
    )
    time_hi = max(cell_ms, np.ceil(time_max_ms / cell_ms) * cell_ms)
    label_step_ms = max(1.0, float(np.ceil(time_hi / 8)))
    if bins is None:
        time_edges = np.arange(0, time_hi + cell_ms / 2, cell_ms)
    else:
        time_edges = np.linspace(0, time_hi, bins + 1)

    style = {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "hatch.linewidth": 0.45,
    }
    dark, mid, light, black = "#244A7C", "#4F81BD", "#A9C7E8", "#202020"

    with plt.rc_context(style):
        fig, axes = plt.subplots(
            2,
            2,
            figsize=(10.2, 5.7),
            gridspec_kw={"width_ratios": (1.55, 1.0)},
        )
        for row, phase in enumerate(PHASES):
            total = np.asarray(samples[phase]["total_ns"]) / 1e6
            comm = np.asarray(samples[phase]["comm_ns"]) / 1e6
            skew = np.asarray(samples[phase]["skew_ns"]) / 1e6
            count = len(total)

            dist = axes[row, 0]
            dist.spines["top"].set_visible(False)
            dist.spines["right"].set_visible(False)
            dist.tick_params(direction="in", length=3, width=0.7)
            dist.set_axisbelow(True)
            dist.grid(
                axis="both",
                color="#D8D8D8",
                linewidth=0.5,
                alpha=0.75,
                zorder=0,
            )
            dist.hist(
                skew,
                bins=time_edges,
                color=light,
                edgecolor=mid,
                alpha=0.72,
                hatch="xx",
                linewidth=0.55,
                label="Skew",
                zorder=2,
            )
            dist.hist(
                comm,
                bins=time_edges,
                color=dark,
                edgecolor=dark,
                alpha=0.48,
                hatch="////",
                linewidth=0.55,
                label="True comm",
                zorder=3,
            )
            dist.hist(
                total,
                bins=time_edges,
                histtype="step",
                color=black,
                linewidth=1.45,
                label="Total",
                zorder=5,
            )
            dist.hist(
                skew,
                bins=time_edges,
                histtype="step",
                color=mid,
                linewidth=1.0,
                zorder=4,
            )
            dist.hist(
                comm,
                bins=time_edges,
                histtype="step",
                color=dark,
                linewidth=1.0,
                zorder=4,
            )
            dist.set_xlim(0, time_hi)
            dist.xaxis.set_major_locator(MultipleLocator(label_step_ms))
            dist.xaxis.set_minor_locator(MultipleLocator(cell_ms))
            dist.grid(which="minor", axis="x", color="#E8E8E8", linewidth=0.4)
            dist.set_xlabel("Time (ms)")
            dist.set_yscale("log")
            dist.set_ylabel("Calls (log scale)")
            dist.legend(
                frameon=False,
                ncol=3,
                loc="upper right",
                handlelength=1.7,
                columnspacing=1.0,
            )
            letter = "ac"[row]
            dist.set_title(
                f"({letter}) {phase} time distributions  [n={count:,}]",
                loc="left",
                fontweight="bold",
                pad=5,
            )

            decomp = axes[row, 1]
            decomp.spines["top"].set_visible(False)
            decomp.spines["right"].set_visible(False)
            decomp.tick_params(direction="in", length=3, width=0.7)
            decomp.set_axisbelow(True)

            order = np.argsort(total)
            percentile = np.linspace(0, 100, count)
            sorted_comm = comm[order]
            sorted_skew = skew[order]
            sorted_total = total[order]
            decomp.fill_between(
                percentile,
                0,
                sorted_comm,
                color=dark,
                alpha=0.78,
                label="True comm",
                linewidth=0,
                zorder=2,
            )
            decomp.fill_between(
                percentile,
                sorted_comm,
                sorted_total,
                color=light,
                alpha=0.90,
                label="Skew",
                linewidth=0,
                zorder=3,
            )
            decomp.plot(
                percentile,
                sorted_total,
                color=black,
                linewidth=1.25,
                label="Total",
                zorder=4,
            )
            decomp.set_xlim(0, 100)
            decomp.set_ylim(0, time_hi)
            decomp.set_xticks((0, 20, 40, 60, 80, 100))
            decomp.yaxis.set_major_locator(MultipleLocator(label_step_ms))
            decomp.yaxis.set_minor_locator(MultipleLocator(cell_ms))
            decomp.grid(
                which="major",
                axis="both",
                color="#D8D8D8",
                linewidth=0.5,
                alpha=0.75,
                zorder=0,
            )
            decomp.grid(
                which="minor",
                axis="y",
                color="#E8E8E8",
                linewidth=0.4,
                zorder=0,
            )
            decomp.set_xlabel("Calls ordered by total time (percentile)")
            decomp.set_ylabel("Time (ms)")
            decomp.legend(
                frameon=False,
                ncol=3,
                loc="upper left",
                handlelength=1.7,
                columnspacing=1.0,
            )
            letter = "bd"[row]
            decomp.set_title(
                f"({letter}) {phase}: Total = True comm + Skew",
                loc="left",
                fontweight="bold",
                pad=5,
            )

        fig.subplots_adjust(
            left=0.075,
            right=0.985,
            bottom=0.105,
            top=0.965,
            wspace=0.24,
            hspace=0.48,
        )
        path = Path(out)
        if path.suffix.lower() != ".png":
            path = Path(str(path) + ".png")
        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)
    return str(path), warnings


def _fmt_ranks(ranks: list[int]) -> str:
    """Compact a sorted rank list: contiguous -> "a-b", else comma list (capped)."""
    if not ranks:
        return ""
    if ranks == list(range(ranks[0], ranks[-1] + 1)) and len(ranks) > 1:
        return f"{ranks[0]}-{ranks[-1]}"
    s = ",".join(map(str, ranks[:6]))
    return s + ("…" if len(ranks) > 6 else "")


def group_ranks(per_rank: dict[int, dict], ep: int) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for rank in per_rank:
        groups[rank // ep].append(rank)
    for g in groups.values():
        g.sort()
    return groups


# =============================================================================
# Report
# =============================================================================


def _phase_stats(per_rank: dict, ranks: list[int], phase: str | None):
    """(times{rank}, bytes{rank}, calls_avg) for one phase; phase=None -> Total (D+C)."""
    times, byts = {}, {}
    calls_sum = 0
    for r in ranks:
        if phase is None:
            times[r] = sum(per_rank[r][p][1] for p in PHASES)
            byts[r] = sum(per_rank[r][p][0] for p in PHASES)
            calls_sum += sum(per_rank[r][p][2] for p in PHASES)
        else:
            times[r] = per_rank[r][phase][1]
            byts[r] = per_rank[r][phase][0]
            calls_sum += per_rank[r][phase][2]
    return times, byts, round(calls_sum / len(ranks))


def print_report(
    per_rank: dict, ep: int, sort: str, per_rank_detail: bool, world: int | None
) -> None:
    groups = group_ranks(per_rank, ep)

    # per-group total-A2A skew, used for ordering + the summary
    def group_total_skew(gid):
        times, _, _ = _phase_stats(per_rank, groups[gid], None)
        return max(times.values()) - min(times.values())

    order = (
        sorted(groups, key=group_total_skew, reverse=True)
        if sort == "skew"
        else sorted(groups)
    )

    header = (
        f"{'group':<6} {'ranks':<12} {'phase':<9} {'calls':>6} {'vol/rank':>10} "
        f"{'floor':>10} {'mean':>10} {'max':>10} {'skew':>10} {'skew%':>6} "
        f"{'BW@floor':>10} {'strag':>6}"
    )
    wtxt = f"world {world}, " if world else ""
    print(
        f"=== DeepEP per-EP-group A2A time & skew ({wtxt}EP group = {ep} ranks, "
        f"{len(groups)} group(s)) ==="
    )
    print(header)
    print("-" * len(header))
    worst = None
    for gid in order:
        ranks = groups[gid]
        rlab = _fmt_ranks(ranks)
        first = True
        for phase in ("Dispatch", "Combine", None):
            times, byts, calls = _phase_stats(per_rank, ranks, phase)
            if not any(times.values()):
                continue
            floor, mx = min(times.values()), max(times.values())
            mean = sum(times.values()) / len(times)
            skew = mx - floor
            strag = max(times, key=times.get)
            vol = sum(byts.values()) / len(ranks)
            name = phase or "Total"
            print(
                f"{('g' + str(gid)) if first else '':<6} {(rlab if first else ''):<12} "
                f"{name:<9} {calls:>6} {human_bytes(vol):>10} {human_ns(floor):>10} "
                f"{human_ns(mean):>10} {human_ns(mx):>10} {human_ns(skew):>10} "
                f"{(skew / mx * 100 if mx else 0):>5.1f}% {human_bw(vol, floor):>10} "
                f"{('r' + str(strag)):>6}"
            )
            if phase is None and (worst is None or skew > worst[1]):
                worst = (gid, skew, skew / mx * 100 if mx else 0, strag)
            first = False
        if per_rank_detail:
            for r in ranks:
                d, c = per_rank[r]["Dispatch"], per_rank[r]["Combine"]
                print(
                    f"       r{r:<5}  D {human_ns(d[1]):>9} ({d[2]}x)   "
                    f"C {human_ns(c[1]):>9} ({c[2]}x)   "
                    f"total {human_ns(d[1] + c[1]):>9}"
                )
    print()
    if worst:
        gid, skew, pct, strag = worst
        print(
            f"worst group by total A2A skew: g{gid} — {human_ns(skew)} "
            f"({pct:.1f}%), straggler r{strag}"
        )


def write_csv(per_rank: dict, ep: int, out: str) -> str:
    """Full per-rank export: rank,group,phase,calls,bytes,gpu_dur_ns."""
    rows = ["rank,group,phase,calls,bytes,gpu_dur_ns"]
    for rank in sorted(per_rank):
        for phase in PHASES:
            b, t, c = per_rank[rank][phase]
            rows.append(f"{rank},{rank // ep},{phase},{c},{b},{t}")
    outp = out if out.endswith(".csv") else out + ".csv"
    Path(outp).write_text("\n".join(rows) + "\n")
    return outp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-EP-group DeepEP dispatch/combine A2A time & skew (DeepEP only)."
    )
    p.add_argument(
        "db", nargs="+", help="One or more per-rank .sqlite profiles (the world)"
    )
    p.add_argument(
        "--ep",
        type=int,
        default=LOCAL_WORLD,
        metavar="N",
        help=f"EP degree = all-to-all group size; consecutive ranks rank//N form one "
        f"EP group (skew is within a group). Default {LOCAL_WORLD}",
    )
    p.add_argument(
        "--world",
        type=int,
        default=None,
        metavar="N",
        help="expected world size (informational; warns if the loaded rank count "
        "differs). Default = number of profiles given",
    )
    p.add_argument("--step-nvtx", default="Optimizer.step", metavar="SUBSTR")
    p.add_argument("--skip-steps", type=int, default=1, metavar="N")
    p.add_argument(
        "--dtype-bytes",
        type=int,
        default=2,
        metavar="N",
        help="bytes/element for DeepEP token volume (no dtype marker); "
        "default 2 (bf16), 1 for fp8, 4 for fp32",
    )
    p.add_argument(
        "--sort",
        choices=("skew", "id"),
        default="skew",
        help="group order: 'skew' (worst total-A2A skew first) or 'id'",
    )
    p.add_argument(
        "--per-rank",
        action="store_true",
        help="also print each rank's dispatch/combine time within its group",
    )
    p.add_argument("--csv", default=None, metavar="OUT", help="write full per-rank CSV")
    p.add_argument(
        "--png",
        default=None,
        metavar="OUT",
        help="write a two-row publication-style PNG: per-phase time distributions "
        "plus a continuous Total = True comm + Skew percentile curve",
    )
    p.add_argument(
        "--hist-bins",
        type=int,
        default=None,
        metavar="N",
        help="override the default 0.25 ms histogram-bin width with N bins",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=min(8, os.cpu_count() or 1),
        metavar="N",
        help="parallel rank-loading processes (default min(8, cpu)); 1 = serial",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    records, durs, nsteps = load_all(
        args.db,
        args.step_nvtx,
        args.skip_steps,
        args.dtype_bytes,
        args.jobs,
    )
    if not records:
        sys.exit("no ranks loaded (no RANK env in the profiles?)")
    if args.world and args.world != len(records):
        print(
            f"warning: --world {args.world} but {len(records)} profiles loaded; "
            f"EP groups with missing ranks will be incomplete",
            file=sys.stderr,
        )
    wmin, wmax = min(durs), max(durs)
    print(
        f"Window: '{args.step_nvtx}' markers={nsteps}, skipping first "
        f"{args.skip_steps} (warmup); post-warmup wall ≈ {human_ns(wmin)}"
        + (f"–{human_ns(wmax)}" if wmax - wmin > 1e6 else "")
        + f" over {len(records)} ranks\n"
    )
    per_rank = accumulate(records)
    if not any(per_rank[r][p][2] for r in per_rank for p in PHASES):
        sys.exit("no DeepEP dispatch/combine kernels found in the window")
    print_report(per_rank, args.ep, args.sort, args.per_rank, args.world)
    if args.csv:
        print(f"\nWrote per-rank CSV -> {write_csv(per_rank, args.ep, args.csv)}")
    if args.png:
        try:
            png, plot_warnings = render_histograms(
                records, args.ep, args.png, args.hist_bins
            )
        except ValueError as error:
            sys.exit(f"cannot render DeepEP histogram: {error}")
        for warning in plot_warnings:
            print(f"warning: {warning}", file=sys.stderr)
        print(f"\nWrote DeepEP per-call histograms -> {png}")
