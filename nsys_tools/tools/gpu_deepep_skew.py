"""gpu-deepep-skew: per-EP-group DeepEP and expert compute time & skew.

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

The MLP and Attention sections group kernels by their exact enclosing
``mlp forward``/``attn forward``, ``mlp backward``/``attn backward``, or
``mlp wgrad``/``attn wgrad`` NVTX instances, measure the earliest-to-latest
GPU kernel span, and align instances across ranks by occurrence within the phase.

Author: yezhengmaolove@gmail.com
"""

import argparse
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path

from .exporter import read_epoch
from .flamegraph import compute_step_windows, detect_steps
from ..utils.comm import CommEvent, load_deepep_in_window
from ..utils.common import get_rank, human_ns, open_db, require_kernel_table
from ..utils.kernel import KERNEL_SQL
from ..utils.nvtx import NvtxIndex

LOCAL_WORLD = 8
CDF_DISPLAY_PERCENTILE = 99.5
PHASES = ("Dispatch", "Combine")
COMPUTE_PHASES = ("Forward", "Dgrad", "Wgrad")
MLP_PHASES = COMPUTE_PHASES
MLP_NVTX_TO_PHASE = {
    "mlp forward": "Forward",
    "mlp backward": "Dgrad",
    "mlp wgrad": "Wgrad",
}
ATTN_NVTX_TO_PHASE = {
    "attn forward": "Forward",
    "attn backward": "Dgrad",
    "attn wgrad": "Wgrad",
}
MLP_LABEL = "Expert MLP"
ATTN_LABEL = "Attention"

MLP_PHASE_COLORS = {
    "Forward": "#6F9FC8",
    "Dgrad": "#E5A064",
    "Wgrad": "#75AF83",
}
ATTN_PHASE_COLORS = {
    "Forward": "#4F81BD",
    "Dgrad": "#C77A53",
    "Wgrad": "#6CA67F",
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


HANDOFF_NVTX_SQL = """
SELECT n.start, n.end, n.globalTid AS tid,
       COALESCE(n.text, s.value) AS name
FROM NVTX_EVENTS n
LEFT JOIN StringIds s ON s.id = n.textId
WHERE n.end IS NOT NULL
  AND (COALESCE(n.text, s.value) IN ("mlp forward", "attn forward")
       OR COALESCE(n.text, s.value) LIKE "FusedDispatch%"
       OR COALESCE(n.text, s.value) LIKE "FusedCombine%")
ORDER BY n.globalTid, n.start, n.end
"""


def load_forward_handoffs(
    conn, ws: int, we: int
) -> dict[tuple[str, int, int], tuple[int, int]]:
    """Map Forward NVTX ranges to their directly adjacent DeepEP range.

    Only semantic adjacency on the same CPU thread is accepted. Dense paths
    therefore remain unmapped instead of being attached to a later MoE call.
    """
    relevant_by_tid = defaultdict(list)
    for row in conn.execute(HANDOFF_NVTX_SQL):
        relevant_by_tid[row["tid"]].append(row)

    expected = {
        "mlp forward": "FusedCombine",
        "attn forward": "FusedDispatch",
    }
    result = {}
    for rows in relevant_by_tid.values():
        for index, row in enumerate(rows[:-1]):
            target_prefix = expected.get(row["name"])
            if target_prefix is None or not (ws < row["start"] <= we):
                continue
            target = rows[index + 1]
            if (
                target["start"] >= row["end"]
                and target["name"].startswith(target_prefix)
            ):
                result[(row["name"], row["start"], row["end"])] = (
                    target["start"],
                    target["end"],
                )
    return result


@dataclass(frozen=True)
class MlpInstance:
    """One exact MLP NVTX occurrence and the span of kernels launched under it."""

    nvtx_start: int
    nvtx_end: int
    gpu_start: int | None
    gpu_end: int | None
    kernels: int
    step: int
    next_comm_call_id: tuple[int, int] | None = None

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


def _window(
    conn, idx: NvtxIndex, needle: str, skip: int
) -> tuple[int, int, int, list[tuple[int, int, int]]]:
    """Return the analysis window and retained one-based training-step windows."""
    steps = detect_steps(idx, needle)
    t_min = conn.execute(TRACE_START_SQL).fetchone()[0]
    if len(steps) <= skip:
        t_max = conn.execute(TRACE_END_SQL).fetchone()[0]
        return t_min, t_max, len(steps), [(t_min, t_max, 1)]
    windows, window_start, window_end = compute_step_windows(steps, t_min, skip)
    retained = [
        (start, end, index + 1)
        for index, (start, end, _) in enumerate(windows)
        if index >= skip
    ]
    return window_start, window_end, len(steps), retained


def load_compute_instances(
    conn,
    idx: NvtxIndex,
    ws: int,
    we: int,
    step_windows: list[tuple[int, int, int]],
    nvtx_to_phase: dict[str, str],
) -> dict[str, list[MlpInstance]]:
    """Collect exact NVTX occurrences and their earliest-to-latest GPU span."""
    scratch: dict[tuple[str, int, int], list[int | None]] = {}
    instance_steps: dict[tuple[str, int, int], int] = {}
    for start, end, name in zip(idx.starts, idx.ends, idx.names):
        phase = nvtx_to_phase.get(name)
        start, end = int(start), int(end)
        if phase is None or not (ws < start <= we):
            continue
        step = next(
            (
                step_index
                for step_start, step_end, step_index in step_windows
                if step_start < start <= step_end
            ),
            None,
        )
        if step is not None:
            key = (phase, start, end)
            scratch[key] = [None, None, 0]
            instance_steps[key] = step

    rows = conn.execute(MLP_KERNEL_SQL, (ws, we)).fetchall()
    stacks = idx.iter_stacks((row["api_start"], row["api_end"]) for row in rows)
    for row, (_, _, stack) in zip(rows, stacks):
        target = None
        for frame in reversed(stack):
            phase = nvtx_to_phase.get(frame.name)
            if phase is not None:
                target = (phase, frame.start, frame.end)
                break
        if target not in scratch:
            continue

        # Wgrad may synchronously launch work on another CUDA stream. Ownership
        # remains defined by the enclosing compute-family NVTX instance.
        slot = scratch[target]
        gpu_start, gpu_end = row["gpu_start"], row["gpu_end"]
        slot[0] = gpu_start if slot[0] is None else min(slot[0], gpu_start)
        slot[1] = gpu_end if slot[1] is None else max(slot[1], gpu_end)
        slot[2] += 1

    result = {phase: [] for phase in COMPUTE_PHASES}
    for (phase, start, end), (gpu_start, gpu_end, kernels) in scratch.items():
        result[phase].append(
            MlpInstance(
                start,
                end,
                gpu_start,
                gpu_end,
                kernels,
                instance_steps[(phase, start, end)],
            )
        )
    for instances in result.values():
        instances.sort(key=lambda item: item.nvtx_start)
    return result


def load_mlp_instances(
    conn, idx: NvtxIndex, ws: int, we: int, step_windows: list[tuple[int, int, int]]
) -> dict[str, list[MlpInstance]]:
    """Collect exact Expert-MLP NVTX occurrences and their GPU spans."""
    return load_compute_instances(
        conn, idx, ws, we, step_windows, MLP_NVTX_TO_PHASE
    )


def load_attn_instances(
    conn, idx: NvtxIndex, ws: int, we: int, step_windows: list[tuple[int, int, int]]
) -> dict[str, list[MlpInstance]]:
    """Collect exact attention NVTX occurrences and their GPU spans."""
    return load_compute_instances(
        conn, idx, ws, we, step_windows, ATTN_NVTX_TO_PHASE
    )


def load_rank(
    db: str, needle: str, skip: int, dtype_bytes: int
) -> tuple:
    """Load DeepEP dispatch/combine events and compute-family spans for one rank.
    Returns (rank, events, mlp, attn, utc_epoch_ns, window_duration_ns, n_steps)."""
    conn = open_db(db)
    require_kernel_table(conn, db)
    rank = get_rank(conn)
    epoch = read_epoch(conn)
    utc_offset = epoch or 0
    idx = NvtxIndex(conn, rank)
    ws, we, nsteps, step_windows = _window(conn, idx, needle, skip)
    events = load_deepep_in_window(conn, idx, ws, we, dtype_bytes)
    mlp = load_mlp_instances(conn, idx, ws, we, step_windows)
    attn = load_attn_instances(conn, idx, ws, we, step_windows)
    handoffs = load_forward_handoffs(conn, ws, we)
    mlp = {
        phase: [
            replace(
                instance,
                gpu_start=(
                    None if instance.gpu_start is None else instance.gpu_start + utc_offset
                ),
                gpu_end=None if instance.gpu_end is None else instance.gpu_end + utc_offset,
                next_comm_call_id=(
                    handoffs.get(("mlp forward", instance.nvtx_start, instance.nvtx_end))
                    if phase == "Forward"
                    else None
                ),
            )
            for instance in instances
        ]
        for phase, instances in mlp.items()
    }
    attn = {
        phase: [
            replace(
                instance,
                gpu_start=(
                    None if instance.gpu_start is None else instance.gpu_start + utc_offset
                ),
                gpu_end=None if instance.gpu_end is None else instance.gpu_end + utc_offset,
                next_comm_call_id=(
                    handoffs.get(("attn forward", instance.nvtx_start, instance.nvtx_end))
                    if phase == "Forward"
                    else None
                ),
            )
            for instance in instances
        ]
        for phase, instances in attn.items()
    }
    conn.close()
    events = [
        event._replace(gpu_start=event.gpu_start + utc_offset) for event in events
    ]
    return rank, events, mlp, attn, epoch, we - ws, nsteps



def load_all(
    dbs: list[str], needle: str, skip: int, dtype_bytes: int, jobs: int
) -> tuple[
    list[tuple[int, list[CommEvent]]],
    dict[int, dict[str, list[MlpInstance]]],
    dict[int, dict[str, list[MlpInstance]]],
    dict[int, int | None],
    list[int],
    int,
]:
    """Load DeepEP events and compute-family spans from every rank."""
    records: list[tuple[int, list[CommEvent]]] = []
    mlp_per_rank: dict[int, dict[str, list[MlpInstance]]] = {}
    attn_per_rank: dict[int, dict[str, list[MlpInstance]]] = {}
    epochs: dict[int, int | None] = {}
    durs: list[int] = []
    nsteps = 0

    def consume(rank, events, mlp, attn, epoch, wdur, nstep):
        nonlocal nsteps
        if rank is None:
            print("  skipped a profile with no RANK env", file=sys.stderr)
            return
        records.append((rank, events))
        mlp_per_rank[rank] = mlp
        attn_per_rank[rank] = attn
        epochs[rank] = epoch
        durs.append(wdur)
        nsteps = max(nsteps, nstep)
        n_dep = sum(1 for e in events if e.op in PHASES)
        n_mlp = sum(len(items) for items in mlp.values())
        n_attn = sum(len(items) for items in attn.values())
        print(
            f"  loaded rank {rank}: {n_dep} DeepEP kernels, {n_mlp} MLP spans, {n_attn} Attention spans",
            file=sys.stderr,
        )

    if jobs <= 1 or len(dbs) == 1:
        for db in dbs:
            consume(*load_rank(db, needle, skip, dtype_bytes))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futs = [ex.submit(load_rank, db, needle, skip, dtype_bytes) for db in dbs]
            for fut in as_completed(futs):
                consume(*fut.result())
    return records, mlp_per_rank, attn_per_rank, epochs, durs, nsteps


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
    call_id: tuple[int, int]
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
                    call_id=event.call_id,
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




@dataclass(frozen=True)
class ComputeCommSample:
    """Compute-duration and UTC-aligned communication-start skew for one pair."""

    source_duration_skew_ns: int
    target_start_skew_ns: int




def _build_compute_comm_samples(
    records: list[tuple[int, list[CommEvent]]],
    compute_per_rank: dict[int, dict[str, list[MlpInstance]]],
    epochs: dict[int, int | None],
    ep: int,
    compute_phase: str,
    target_phase: str,
    source_label: str,
) -> tuple[list[ComputeCommSample], list[str]]:
    """Pair compute with its directly adjacent NVTX communication range.

    Rank-local NVTX adjacency establishes causality. Cross-rank occurrences are
    then ordered by the target range first-kernel UTC start within each step.
    No sequence field or fixed time bin is used.
    """
    calls = _logical_calls_by_rank(records)
    calls_by_id = {
        rank: {call.call_id: call for call in rank_calls}
        for rank, rank_calls in calls.items()
    }
    samples = []
    warnings = []
    relation = f"{source_label} {compute_phase} -> {target_phase}"

    def count_text(counts):
        values = set(counts.values())
        return str(next(iter(values))) if len(values) == 1 else str(counts)

    for gid, ranks in sorted(group_ranks(compute_per_rank, ep).items()):
        missing_epochs = [rank for rank in ranks if epochs.get(rank) is None]
        if missing_epochs:
            warnings.append(
                f"g{gid}: skipped {relation}; no UTC session epoch for ranks "
                + ",".join(map(str, missing_epochs))
            )
            continue
        if len(ranks) != ep or any(rank not in calls_by_id for rank in ranks):
            warnings.append(f"g{gid}: skipped incomplete {relation} group")
            continue

        per_rank_step = {}
        ignored_counts = {}
        missing_kernel_counts = {}
        duplicate_target_counts = {}
        empty_compute_counts = {}
        for rank in ranks:
            grouped = defaultdict(list)
            ignored = missing_kernel = duplicate_target = empty_compute = 0
            used_call_ids = set()
            for instance in compute_per_rank[rank][compute_phase]:
                if instance.gpu_start is None or instance.gpu_end is None:
                    empty_compute += 1
                    continue
                call_id = instance.next_comm_call_id
                if call_id is None:
                    ignored += 1
                    continue
                call = calls_by_id[rank].get(call_id)
                if call is None or call.phase != target_phase:
                    missing_kernel += 1
                    continue
                if call_id in used_call_ids:
                    duplicate_target += 1
                    continue
                used_call_ids.add(call_id)
                grouped[instance.step].append((call.start_ns, instance, call))
            for items in grouped.values():
                items.sort(key=lambda item: item[0])
            per_rank_step[rank] = grouped
            ignored_counts[rank] = ignored
            missing_kernel_counts[rank] = missing_kernel
            duplicate_target_counts[rank] = duplicate_target
            empty_compute_counts[rank] = empty_compute

        steps = sorted(
            {step for grouped in per_rank_step.values() for step in grouped}
        )
        mismatch_steps = 0
        group_samples = 0
        for step in steps:
            counts = {
                rank: len(per_rank_step[rank].get(step, [])) for rank in ranks
            }
            if len(set(counts.values())) != 1:
                mismatch_steps += 1
                warnings.append(
                    f"g{gid} {relation} S{step}: unequal NVTX-bound counts {counts}; "
                    "step skipped"
                )
                continue
            count = next(iter(counts.values()), 0)
            for occurrence in range(count):
                matched = {
                    rank: per_rank_step[rank][step][occurrence] for rank in ranks
                }
                source_duration = [
                    matched[rank][1].gpu_end - matched[rank][1].gpu_start
                    for rank in ranks
                ]
                target_start = [matched[rank][2].start_ns for rank in ranks]
                samples.append(
                    ComputeCommSample(
                        source_duration_skew_ns=(
                            max(source_duration) - min(source_duration)
                        ),
                        target_start_skew_ns=(
                            max(target_start) - min(target_start)
                        ),
                    )
                )
                group_samples += 1

        print(
            f"  {relation} g{gid}: paired={group_samples}, "
            f"dense/non-adjacent={count_text(ignored_counts)}, "
            f"empty-compute={count_text(empty_compute_counts)}, "
            f"missing-target-kernel={count_text(missing_kernel_counts)}, "
            f"duplicate-target={count_text(duplicate_target_counts)}, "
            f"count-mismatch-steps={mismatch_steps}",
            file=sys.stderr,
        )
        if (
            any(empty_compute_counts.values())
            or any(missing_kernel_counts.values())
            or any(duplicate_target_counts.values())
        ):
            warnings.append(
                f"g{gid} {relation}: missing target kernels "
                f"{missing_kernel_counts}, empty compute {empty_compute_counts}, "
                f"duplicate targets {duplicate_target_counts}"
            )
    return samples, warnings


def build_mlp_combine_samples(
    records: list[tuple[int, list[CommEvent]]],
    mlp_per_rank: dict[int, dict[str, list[MlpInstance]]],
    epochs: dict[int, int | None],
    ep: int,
) -> tuple[list[ComputeCommSample], list[str]]:
    return _build_compute_comm_samples(
        records, mlp_per_rank, epochs, ep, "Forward", "Combine", MLP_LABEL
    )


def build_attn_dispatch_samples(
    records: list[tuple[int, list[CommEvent]]],
    attn_per_rank: dict[int, dict[str, list[MlpInstance]]],
    epochs: dict[int, int | None],
    ep: int,
) -> tuple[list[ComputeCommSample], list[str]]:
    return _build_compute_comm_samples(
        records, attn_per_rank, epochs, ep, "Forward", "Dispatch", ATTN_LABEL
    )


def render_histograms(
    records: list[tuple[int, list[CommEvent]]],
    mlp_per_rank: dict[int, dict[str, list[MlpInstance]]],
    attn_per_rank: dict[int, dict[str, list[MlpInstance]]],
    epochs: dict[int, int | None],
    ep: int,
    out: str,
    bins: int | None = None,
) -> tuple[str, list[str]]:
    """Write combined DeepEP distributions and Expert-MLP / Attention skew diagnostics."""
    if bins is not None and bins < 1:
        raise ValueError("histogram bins must be positive")
    samples, warnings = build_histogram_samples(records, ep)
    if not any(samples[phase]["total_ns"] for phase in PHASES):
        raise ValueError("no complete, cross-rank DeepEP calls available for plotting")
    mlp_samples, mlp_warnings = build_mlp_diagnostic_samples(mlp_per_rank, ep)
    attn_samples, attn_warnings = build_attn_diagnostic_samples(attn_per_rank, ep)
    mlp_combine_samples, correlation_warnings = build_mlp_combine_samples(
        records, mlp_per_rank, epochs, ep
    )
    attn_dispatch_samples, attn_correlation_warnings = build_attn_dispatch_samples(
        records, attn_per_rank, epochs, ep
    )
    warnings.extend(mlp_warnings)
    warnings.extend(attn_warnings)
    warnings.extend(correlation_warnings)
    warnings.extend(attn_correlation_warnings)
    compute_rows = []
    has_mlp = any(mlp_samples[phase] for phase in MLP_PHASES)
    has_attn = any(attn_samples[phase] for phase in COMPUTE_PHASES)
    if has_mlp:
        compute_rows.append((MLP_LABEL, mlp_samples, MLP_PHASE_COLORS))
    if has_attn:
        compute_rows.append((ATTN_LABEL, attn_samples, ATTN_PHASE_COLORS))

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
    cdf_max_ms = max(
        np.percentile(
            np.asarray(samples[phase]["total_ns"]) / 1e6,
            CDF_DISPLAY_PERCENTILE,
        )
        for phase in PHASES
    )
    cdf_hi = max(cell_ms, np.ceil(cdf_max_ms / cell_ms) * cell_ms)
    cdf_label_step_ms = max(1.0, float(np.ceil(cdf_hi / 8)))
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
    letters = "abcdefghijklmnopqrstuvwxyz"

    with plt.rc_context(style):
        fig = plt.figure(figsize=(16.2, 7.2))
        outer = fig.add_gridspec(
            2,
            4,
            height_ratios=(1.08, 0.92),
            left=0.055,
            right=0.985,
            bottom=0.085,
            top=0.96,
            wspace=0.42,
            hspace=0.40,
        )
        dist_axes = [
            fig.add_subplot(outer[0, 0]),
            fig.add_subplot(outer[0, 1]),
        ]
        cdf_axes = [
            fig.add_subplot(outer[1, 0]),
            fig.add_subplot(outer[1, 1]),
        ]
        for row, phase in enumerate(PHASES):
            total = np.asarray(samples[phase]["total_ns"]) / 1e6
            comm = np.asarray(samples[phase]["comm_ns"]) / 1e6
            skew = np.asarray(samples[phase]["skew_ns"]) / 1e6
            count = len(total)

            dist = dist_axes[row]
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
            letter = letters[row]
            dist.set_title(
                f"({letter}) {phase} time distributions  [n={count:,}]",
                loc="left",
                fontweight="bold",
                pad=5,
            )

            decomp = cdf_axes[row]
            decomp.spines["top"].set_visible(False)
            decomp.spines["right"].set_visible(False)
            decomp.tick_params(direction="in", length=3, width=0.7)
            decomp.set_axisbelow(True)

            percentile = np.arange(1, count + 1) / count * 100
            sorted_comm = np.sort(comm)
            sorted_skew = np.sort(skew)
            sorted_total = np.sort(total)
            decomp.plot(
                sorted_comm,
                percentile,
                color=dark,
                linewidth=1.35,
                label="True comm",
                drawstyle="steps-post",
                zorder=3,
            )
            decomp.plot(
                sorted_skew,
                percentile,
                color=mid,
                linewidth=1.35,
                label="Skew",
                drawstyle="steps-post",
                zorder=2,
            )
            decomp.plot(
                sorted_total,
                percentile,
                color=black,
                linewidth=1.5,
                label="Total",
                drawstyle="steps-post",
                zorder=4,
            )
            decomp.set_xlim(0, cdf_hi)
            decomp.set_ylim(0, 100)
            decomp.set_yticks((0, 20, 40, 60, 80, 100))
            decomp.xaxis.set_major_locator(MultipleLocator(cdf_label_step_ms))
            decomp.xaxis.set_minor_locator(MultipleLocator(cell_ms))
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
                axis="x",
                color="#E8E8E8",
                linewidth=0.4,
                zorder=0,
            )
            decomp.set_xlabel("Time (ms)")
            decomp.set_ylabel("Cumulative calls (%)")
            decomp.text(
                0.98,
                0.96,
                f"X capped at Total P{CDF_DISPLAY_PERCENTILE:g}",
                transform=decomp.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color="#555555",
            )
            decomp.legend(
                frameon=False,
                ncol=3,
                loc="lower right",
                handlelength=1.7,
                columnspacing=1.0,
            )
            letter = letters[row + 2]
            decomp.set_title(
                f"({letter}) {phase} cumulative",
                loc="left",
                fontweight="bold",
                pad=5,
            )

        for compute_index, (label, samples_family, phase_colors) in enumerate(
            compute_rows
        ):
            cumulative_ax = fig.add_subplot(outer[1, compute_index + 2])
            plot_compute_cumulative(
                cumulative_ax,
                samples_family,
                phase_colors,
                letters[compute_index + 4],
                label,
            )

        mlp_correlation_ax = fig.add_subplot(outer[0, 2])
        attn_correlation_ax = fig.add_subplot(outer[0, 3])
        plot_utc_start_correlation(
            mlp_correlation_ax,
            mlp_combine_samples,
            letters[6],
            "Expert MLP Forward",
            "Combine",
            "#244A7C",
        )
        plot_utc_start_correlation(
            attn_correlation_ax,
            attn_dispatch_samples,
            letters[7],
            "Attention Forward",
            "Dispatch",
            "#C77A53",
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


def build_mlp_summaries(
    mlp_per_rank: dict[int, dict[str, list[MlpInstance]]], ep: int
) -> tuple[dict[int, dict[str, MlpPhaseSummary]], list[str]]:
    """Align MLP instances by phase occurrence and summarize complete EP groups."""
    groups = group_ranks(mlp_per_rank, ep)
    summaries: dict[int, dict[str, MlpPhaseSummary]] = defaultdict(dict)
    warnings: list[str] = []

    for gid, ranks in sorted(groups.items()):
        if len(ranks) != ep:
            warnings.append(
                f"g{gid}: skipped incomplete MLP EP group ({len(ranks)}/{ep} ranks)"
            )
            continue
        for phase in MLP_PHASES:
            per_rank = {rank: mlp_per_rank[rank][phase] for rank in ranks}
            counts = {rank: len(items) for rank, items in per_rank.items()}
            max_count = max(counts.values(), default=0)
            if not max_count:
                continue

            occurrence_stats = []
            skipped = 0
            for occurrence in range(max_count):
                spans = {}
                for rank in ranks:
                    items = per_rank[rank]
                    if occurrence >= len(items) or items[occurrence].span is None:
                        break
                    spans[rank] = items[occurrence].span
                if len(spans) != len(ranks):
                    skipped += 1
                    continue
                floor = min(spans.values())
                maximum = max(spans.values())
                skew = maximum - floor
                occurrence_stats.append(
                    {
                        "occurrence": occurrence,
                        "spans": spans,
                        "floor": floor,
                        "mean": sum(spans.values()) / len(spans),
                        "maximum": maximum,
                        "skew": skew,
                        "skew_pct": skew / maximum * 100 if maximum else 0.0,
                        "straggler": max(spans, key=spans.get),
                    }
                )

            if len(set(counts.values())) > 1 or skipped:
                count_text = ", ".join(f"r{rank}={counts[rank]}" for rank in ranks)
                warnings.append(
                    f"g{gid} {phase}: occurrence counts [{count_text}], "
                    f"skipped {skipped} incomplete/empty instance(s)"
                )
            if not occurrence_stats:
                continue

            n = len(occurrence_stats)
            worst = max(occurrence_stats, key=lambda item: item["skew"])
            summaries[gid][phase] = MlpPhaseSummary(
                occurrences=n,
                floor=sum(item["floor"] for item in occurrence_stats) / n,
                mean=sum(item["mean"] for item in occurrence_stats) / n,
                maximum=sum(item["maximum"] for item in occurrence_stats) / n,
                skew=sum(item["skew"] for item in occurrence_stats) / n,
                skew_pct=sum(item["skew_pct"] for item in occurrence_stats) / n,
                worst_occurrence=worst["occurrence"],
                worst_skew=worst["skew"],
                worst_skew_pct=worst["skew_pct"],
                worst_straggler=worst["straggler"],
                per_rank_mean={
                    rank: sum(item["spans"][rank] for item in occurrence_stats) / n
                    for rank in ranks
                },
            )
    return dict(summaries), warnings


def print_mlp_report(
    mlp_per_rank: dict[int, dict[str, list[MlpInstance]]],
    ep: int,
    sort: str,
    per_rank_detail: bool,
    world: int | None,
) -> None:
    """Print occurrence-aligned MLP span/skew statistics for complete EP groups."""
    summaries, warnings = build_mlp_summaries(mlp_per_rank, ep)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    groups = group_ranks(mlp_per_rank, ep)
    gids = [gid for gid in groups if gid in summaries]
    if sort == "skew":
        gids.sort(
            key=lambda gid: max(
                (summary.worst_skew for summary in summaries[gid].values()),
                default=0,
            ),
            reverse=True,
        )
    else:
        gids.sort()

    wtxt = f"world {world}, " if world else ""
    print(
        f"=== Expert MLP per-EP-group GPU span & skew ({wtxt}EP group = {ep} ranks, "
        f"{len(gids)} complete group(s)) ==="
    )
    header = (
        f"{'group':<6} {'ranks':<12} {'phase':<9} {'inst':>6} {'avg-floor':>10} "
        f"{'avg-mean':>10} {'avg-max':>10} {'avg-skew':>10} {'skew%':>6} "
        f"{'worst':>10} {'occ/strag':>11}"
    )
    print(header)
    print("-" * len(header))

    global_worst = None
    for gid in gids:
        ranks = groups[gid]
        first = True
        for phase in MLP_PHASES:
            summary = summaries[gid].get(phase)
            if summary is None:
                continue
            print(
                f"{('g' + str(gid)) if first else '':<6} "
                f"{(_fmt_ranks(ranks) if first else ''):<12} {phase:<9} "
                f"{summary.occurrences:>6} {human_ns(summary.floor):>10} "
                f"{human_ns(summary.mean):>10} {human_ns(summary.maximum):>10} "
                f"{human_ns(summary.skew):>10} {summary.skew_pct:>5.1f}% "
                f"{human_ns(summary.worst_skew):>10} "
                f"#{summary.worst_occurrence + 1}/r{summary.worst_straggler:<5}"
            )
            item = (summary.worst_skew, gid, phase, summary)
            if global_worst is None or item[0] > global_worst[0]:
                global_worst = item
            first = False
        if per_rank_detail:
            for rank in ranks:
                parts = []
                for phase in MLP_PHASES:
                    summary = summaries[gid].get(phase)
                    if summary is not None:
                        parts.append(
                            f"{phase[0]} {human_ns(summary.per_rank_mean[rank])}"
                        )
                print(f"       r{rank:<5}  " + "   ".join(parts))

    if not gids:
        print("no complete cross-rank MLP occurrences found")
    elif global_worst:
        skew, gid, phase, summary = global_worst
        print(
            f"worst MLP instance: g{gid} {phase} #{summary.worst_occurrence + 1} — "
            f"{human_ns(skew)} ({summary.worst_skew_pct:.1f}%), "
            f"straggler r{summary.worst_straggler}"
        )



def _build_compute_diagnostic_samples(
    per_rank: dict[int, dict[str, list[MlpInstance]]],
    ep: int,
    family: str,
) -> tuple[
    dict[str, dict[int, dict[int, list[tuple[int, int, int, int]]]]],
    list[str],
]:
    """Align compute instances by (EP group, training step, phase occurrence)."""
    samples = {phase: defaultdict(dict) for phase in COMPUTE_PHASES}
    warnings = []
    for gid, ranks in sorted(group_ranks(per_rank, ep).items()):
        if len(ranks) != ep:
            warnings.append(
                f"g{gid}: skipped incomplete {family} EP group "
                f"({len(ranks)}/{ep} ranks)"
            )
            continue
        for phase in COMPUTE_PHASES:
            per_rank_step = {}
            for rank in ranks:
                grouped = defaultdict(list)
                for instance in per_rank[rank][phase]:
                    grouped[instance.step].append(instance)
                per_rank_step[rank] = grouped
            steps = sorted(
                {
                    step
                    for grouped in per_rank_step.values()
                    for step in grouped
                }
            )
            for step in steps:
                rank_items = {
                    rank: per_rank_step[rank].get(step, []) for rank in ranks
                }
                counts = {rank: len(items) for rank, items in rank_items.items()}
                max_count = max(counts.values(), default=0)
                complete = []
                skipped = 0
                for occurrence in range(max_count):
                    spans = []
                    for rank in ranks:
                        items = rank_items[rank]
                        if occurrence >= len(items) or items[occurrence].span is None:
                            break
                        spans.append(items[occurrence].span)
                    if len(spans) != len(ranks):
                        skipped += 1
                        continue
                    fastest = min(spans)
                    slowest = max(spans)
                    complete.append(
                        (occurrence, fastest, slowest, slowest - fastest)
                    )
                samples[phase][gid][step] = complete
                count_text = ", ".join(f"r{rank}={counts[rank]}" for rank in ranks)
                print(
                    f"  {family} g{gid} {phase} S{step}: "
                    f"complete={len(complete)} [{count_text}]",
                    file=sys.stderr,
                )
                if len(set(counts.values())) != 1:
                    warnings.append(
                        f"{family} g{gid} {phase} S{step}: occurrence counts "
                        f"differ [{count_text}]"
                    )
                if skipped:
                    warnings.append(
                        f"{family} g{gid} {phase} S{step}: skipped {skipped} "
                        "incomplete/empty instance(s)"
                    )
    return samples, warnings


def build_mlp_diagnostic_samples(
    mlp_per_rank: dict[int, dict[str, list[MlpInstance]]], ep: int
) -> tuple[
    dict[str, dict[int, dict[int, list[tuple[int, int, int, int]]]]],
    list[str],
]:
    return _build_compute_diagnostic_samples(mlp_per_rank, ep, MLP_LABEL)


def _flatten_compute_phase(
    samples: dict[
        str, dict[int, dict[int, list[tuple[int, int, int, int]]]]
    ],
    phase: str,
) -> list[tuple[int, int, int, int]]:
    """Concatenate all retained steps in chronological occurrence order."""
    return [
        item
        for gid, steps in sorted(samples[phase].items())
        for _, values in sorted(steps.items())
        for item in values
    ]


def plot_utc_start_correlation(
    ax,
    samples: list[ComputeCommSample],
    letter: str,
    source_label: str,
    target_label: str,
    color: str,
) -> None:
    """Plot compute-duration skew against UTC-aligned communication-start skew."""
    import numpy as np

    if not samples:
        ax.text(
            0.5,
            0.5,
            f"No unambiguous {source_label} -> {target_label} pairs",
            ha="center",
        )
        ax.set_axis_off()
        return

    x = np.asarray([sample.source_duration_skew_ns for sample in samples]) / 1e3
    y = np.asarray([sample.target_start_skew_ns for sample in samples]) / 1e3
    x_cap = float(np.percentile(x, CDF_DISPLAY_PERCENTILE))
    y_cap = float(np.percentile(y, CDF_DISPLAY_PERCENTILE))
    visible = (x <= x_cap) & (y <= y_cap)
    xv, yv = x[visible], y[visible]

    def average_ranks(values):
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(len(values), dtype=float)
        sorted_values = values[order]
        begin = 0
        while begin < len(values):
            end = begin + 1
            while end < len(values) and sorted_values[end] == sorted_values[begin]:
                end += 1
            ranks[order[begin:end]] = (begin + end - 1) / 2
            begin = end
        return ranks

    rho = (
        float(np.corrcoef(average_ranks(xv), average_ranks(yv))[0, 1])
        if len(xv) >= 2 and np.ptp(xv) > 0 and np.ptp(yv) > 0
        else float("nan")
    )
    slope = (
        float(np.polyfit(xv, yv, 1)[0])
        if len(xv) >= 2 and np.ptp(xv) > 0
        else float("nan")
    )
    ax.scatter(
        xv,
        yv,
        s=8,
        marker="o",
        color=color,
        alpha=0.34,
        linewidths=0,
        rasterized=True,
        label=f"rho={rho:.2f}, slope={slope:.2f}, n={len(xv):,}",
    )
    axis_cap = max(1.0, x_cap, y_cap)
    ax.plot(
        (0, axis_cap),
        (0, axis_cap),
        color="#333333",
        linewidth=0.9,
        linestyle="--",
        label="y = x",
    )
    ax.set_xlim(0, axis_cap)
    ax.set_ylim(0, axis_cap)
    ax.set_aspect("equal", adjustable="box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", length=3, width=0.7)
    ax.grid(color="#D8D8D8", linewidth=0.45, alpha=0.7)
    ax.set_xlabel(f"{source_label} duration skew (us)")
    ax.set_ylabel(f"{target_label} start skew (us, UTC aligned)")
    ax.set_title(
        f"({letter}) {source_label} duration vs {target_label} start",
        loc="left",
        fontweight="bold",
        pad=5,
    )
    ax.text(
        0.99,
        0.03,
        f"Shared axes capped at max P{CDF_DISPLAY_PERCENTILE:g}; UTC aligned",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="#555555",
    )
    ax.legend(frameon=False, loc="upper left", ncol=1)


def plot_compute_cumulative(
    ax,
    samples: dict[
        str, dict[int, dict[int, list[tuple[int, int, int, int]]]]
    ],
    phase_colors: dict[str, str],
    letter: str,
    title_prefix: str,
) -> None:
    """Plot empirical cumulative curves from every raw compute-skew sample."""
    import numpy as np

    curves = []
    for phase in samples:
        items = _flatten_compute_phase(samples, phase)
        if not items:
            continue
        curve = np.sort(np.asarray([item[3] / 1e3 for item in items]))
        percentile = np.arange(1, len(curve) + 1) / len(curve) * 100
        ax.step(
            curve,
            percentile,
            where="post",
            color=phase_colors[phase],
            linewidth=1.2,
            label=phase,
        )
        curves.append(curve)

    if curves:
        x_hi = max(
            float(np.percentile(curve, CDF_DISPLAY_PERCENTILE))
            for curve in curves
        )
        ax.set_xlim(0, max(1.0, x_hi))
    ax.set_ylim(0, 100)
    ax.set_yticks((0, 25, 50, 75, 100))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="in", length=3, width=0.7)
    ax.grid(color="#D8D8D8", linewidth=0.45, alpha=0.7)
    ax.set_xlabel("Cross-rank skew (us)")
    ax.set_ylabel("Cumulative calls (%)")
    ax.set_title(
        f"({letter}) {title_prefix} cumulative",
        loc="left",
        fontweight="bold",
        fontsize=9,
        pad=5,
    )
    ax.text(
        0.98,
        0.04,
        f"X capped at P{CDF_DISPLAY_PERCENTILE:g}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#555555",
    )
    ax.legend(
        frameon=False,
        ncol=1,
        loc="upper left",
        handlelength=1.4,
        labelspacing=0.25,
    )


def build_attn_summaries(
    attn_per_rank: dict[int, dict[str, list[MlpInstance]]], ep: int
) -> tuple[dict[int, dict[str, MlpPhaseSummary]], list[str]]:
    """Align attention instances by phase occurrence and summarize complete EP groups."""
    groups = group_ranks(attn_per_rank, ep)
    summaries: dict[int, dict[str, MlpPhaseSummary]] = defaultdict(dict)
    warnings: list[str] = []

    for gid, ranks in sorted(groups.items()):
        if len(ranks) != ep:
            warnings.append(
                f"g{gid}: skipped incomplete Attention EP group ({len(ranks)}/{ep} ranks)"
            )
            continue
        for phase in COMPUTE_PHASES:
            per_rank = {rank: attn_per_rank[rank][phase] for rank in ranks}
            counts = {rank: len(items) for rank, items in per_rank.items()}
            max_count = max(counts.values(), default=0)
            if not max_count:
                continue

            occurrence_stats = []
            skipped = 0
            for occurrence in range(max_count):
                spans = {}
                for rank in ranks:
                    items = per_rank[rank]
                    if occurrence >= len(items) or items[occurrence].span is None:
                        break
                    spans[rank] = items[occurrence].span
                if len(spans) != len(ranks):
                    skipped += 1
                    continue
                floor = min(spans.values())
                maximum = max(spans.values())
                skew = maximum - floor
                occurrence_stats.append(
                    {
                        "occurrence": occurrence,
                        "spans": spans,
                        "floor": floor,
                        "mean": sum(spans.values()) / len(spans),
                        "maximum": maximum,
                        "skew": skew,
                        "skew_pct": skew / maximum * 100 if maximum else 0.0,
                        "straggler": max(spans, key=spans.get),
                    }
                )

            if len(set(counts.values())) > 1 or skipped:
                count_text = ", ".join(f"r{rank}={counts[rank]}" for rank in ranks)
                warnings.append(
                    f"g{gid} {phase}: occurrence counts [{count_text}], "
                    f"skipped {skipped} incomplete/empty instance(s)"
                )
            if not occurrence_stats:
                continue

            n = len(occurrence_stats)
            worst = max(occurrence_stats, key=lambda item: item["skew"])
            summaries[gid][phase] = MlpPhaseSummary(
                occurrences=n,
                floor=sum(item["floor"] for item in occurrence_stats) / n,
                mean=sum(item["mean"] for item in occurrence_stats) / n,
                maximum=sum(item["maximum"] for item in occurrence_stats) / n,
                skew=sum(item["skew"] for item in occurrence_stats) / n,
                skew_pct=sum(item["skew_pct"] for item in occurrence_stats) / n,
                worst_occurrence=worst["occurrence"],
                worst_skew=worst["skew"],
                worst_skew_pct=worst["skew_pct"],
                worst_straggler=worst["straggler"],
                per_rank_mean={
                    rank: sum(item["spans"][rank] for item in occurrence_stats) / n
                    for rank in ranks
                },
            )
    return dict(summaries), warnings


def print_attn_report(
    attn_per_rank: dict[int, dict[str, list[MlpInstance]]],
    ep: int,
    sort: str,
    per_rank_detail: bool,
    world: int | None,
) -> None:
    """Print occurrence-aligned Attention span/skew statistics for complete EP groups."""
    summaries, warnings = build_attn_summaries(attn_per_rank, ep)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    groups = group_ranks(attn_per_rank, ep)
    gids = [gid for gid in groups if gid in summaries]
    if sort == "skew":
        gids.sort(
            key=lambda gid: max(
                (summary.worst_skew for summary in summaries[gid].values()),
                default=0,
            ),
            reverse=True,
        )
    else:
        gids.sort()

    wtxt = f"world {world}, " if world else ""
    print(
        f"=== Attention per-EP-group GPU span & skew ({wtxt}EP group = {ep} ranks, "
        f"{len(gids)} complete group(s)) ==="
    )
    header = (
        f"{'group':<6} {'ranks':<12} {'phase':<9} {'inst':>6} {'avg-floor':>10} "
        f"{'avg-mean':>10} {'avg-max':>10} {'avg-skew':>10} {'skew%':>6} "
        f"{'worst':>10} {'occ/strag':>11}"
    )
    print(header)
    print("-" * len(header))

    global_worst = None
    for gid in gids:
        ranks = groups[gid]
        first = True
        for phase in COMPUTE_PHASES:
            summary = summaries[gid].get(phase)
            if summary is None:
                continue
            print(
                f"{('g' + str(gid)) if first else '':<6} "
                f"{(_fmt_ranks(ranks) if first else ''):<12} {phase:<9} "
                f"{summary.occurrences:>6} {human_ns(summary.floor):>10} "
                f"{human_ns(summary.mean):>10} {human_ns(summary.maximum):>10} "
                f"{human_ns(summary.skew):>10} {summary.skew_pct:>5.1f}% "
                f"{human_ns(summary.worst_skew):>10} "
                f"#{summary.worst_occurrence + 1}/r{summary.worst_straggler:<5}"
            )
            item = (summary.worst_skew, gid, phase, summary)
            if global_worst is None or item[0] > global_worst[0]:
                global_worst = item
            first = False
        if per_rank_detail:
            for rank in ranks:
                parts = []
                for phase in COMPUTE_PHASES:
                    summary = summaries[gid].get(phase)
                    if summary is not None:
                        parts.append(
                            f"{phase[0]} {human_ns(summary.per_rank_mean[rank])}"
                        )
                print(f"       r{rank:<5}  " + "   ".join(parts))

    if not gids:
        print("no complete cross-rank Attention occurrences found")
    elif global_worst:
        skew, gid, phase, summary = global_worst
        print(
            f"worst Attention instance: g{gid} {phase} #{summary.worst_occurrence + 1} — "
            f"{human_ns(skew)} ({summary.worst_skew_pct:.1f}%), "
            f"straggler r{summary.worst_straggler}"
        )


def build_attn_diagnostic_samples(
    attn_per_rank: dict[int, dict[str, list[MlpInstance]]], ep: int
) -> tuple[
    dict[str, dict[int, dict[int, list[tuple[int, int, int, int]]]]],
    list[str],
]:
    return _build_compute_diagnostic_samples(attn_per_rank, ep, ATTN_LABEL)


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
        description="Per-EP-group DeepEP A2A and Expert MLP/Attention GPU span & skew."
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
    p.add_argument(
        "--step-nvtx",
        default="Optimizer.step#TensorParallelMuon.step",
        metavar="SUBSTR",
        help="training-step end marker substring; default TensorParallelMuon step",
    )
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
        help="also print each rank's communication and average MLP/Attention spans",
    )
    p.add_argument("--csv", default=None, metavar="OUT", help="write full per-rank CSV")
    p.add_argument(
        "--png",
        default=None,
        metavar="OUT",
        help="write one combined PNG: DeepEP distributions/decomposition plus "
        "Expert MLP/Attention skew CDFs and UTC-aligned communication-start correlations",
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
    records, mlp_per_rank, attn_per_rank, epochs, durs, nsteps = load_all(
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
    has_deepep = any(per_rank[r][p][2] for r in per_rank for p in PHASES)
    has_mlp = any(
        instance.span is not None
        for phases in mlp_per_rank.values()
        for instances in phases.values()
        for instance in instances
    )
    has_attn = any(
        instance.span is not None
        for phases in attn_per_rank.values()
        for instances in phases.values()
        for instance in instances
    )
    if not has_deepep and not has_mlp and not has_attn:
        sys.exit("no DeepEP kernels, MLP spans, or Attention spans found in the window")

    printed_section = False
    if has_deepep:
        print_report(per_rank, args.ep, args.sort, args.per_rank, args.world)
        printed_section = True
    else:
        print("warning: no DeepEP dispatch/combine kernels found", file=sys.stderr)

    if has_mlp:
        if printed_section:
            print()
        print_mlp_report(mlp_per_rank, args.ep, args.sort, args.per_rank, args.world)
        printed_section = True
    else:
        print("warning: no MLP NVTX spans found", file=sys.stderr)

    if has_attn:
        if printed_section:
            print()
        print_attn_report(attn_per_rank, args.ep, args.sort, args.per_rank, args.world)
        printed_section = True
    else:
        print("warning: no Attention NVTX spans found", file=sys.stderr)

    if args.csv:
        print(f"\nWrote per-rank CSV -> {write_csv(per_rank, args.ep, args.csv)}")
    if args.png:
        try:
            png, plot_warnings = render_histograms(
                records,
                mlp_per_rank,
                attn_per_rank,
                epochs,
                args.ep,
                args.png,
                args.hist_bins,
            )
        except ValueError as error:
            sys.exit(f"cannot render combined DeepEP/MLP/Attention figure: {error}")
        for warning in plot_warnings:
            print(f"warning: {warning}", file=sys.stderr)
        print(f"\nWrote combined DeepEP/MLP/Attention diagnostics -> {png}")
