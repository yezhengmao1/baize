"""Match AllReduce NVTX markers to GPU kernels and report launch skew."""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .exporter import read_epoch
from ..utils.common import get_rank, human_ns, open_db, require_kernel_table
from ..utils.nvtx import NvtxIndex


ALLREDUCE_RE = re.compile(
    r"^AllReduce:comm=(?P<comm>0x[0-9a-fA-F]+):seq=(?P<seq>\d+)"
    r":algo=(?P<algo>\w+):proto=(?P<proto>\w+):dtype=(?P<dtype>\w+)"
    r":count=(?P<count>\d+)",
    re.IGNORECASE,
)
MARK_SQL = """
SELECT n.start, n.globalTid AS tid, COALESCE(n.text, s.value) AS text
FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id = n.textId
WHERE COALESCE(n.text, s.value) LIKE 'AllReduce:comm=%'
ORDER BY n.start
"""
LAUNCH_SQL = """
SELECT r.start AS api_start, r.end AS api_end, r.globalTid AS tid,
       k.start AS gpu_start, k.end AS gpu_end
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId
JOIN StringIds sa ON sa.id = r.nameId
JOIN StringIds sk ON sk.id = k.shortName
WHERE sa.value LIKE '%LaunchKernel%'
  AND sk.value LIKE 'ncclDevKernel_AllReduce%'
ORDER BY r.start
"""


@dataclass(frozen=True)
class Marker:
    start: int
    tid: int
    comm: str
    seq: int
    algo: str
    proto: str
    dtype: str
    count: int


@dataclass(frozen=True)
class AllReduceKernel:
    rank: int
    comm: str
    seq: int
    algo: str
    proto: str
    dtype: str
    count: int
    gpu_start: int
    gpu_end: int
    epoch: int | None


@dataclass(frozen=True)
class CollectiveSample:
    comm: str
    seq: int
    nranks: int
    member_ranks: tuple[int, ...]
    earliest_rank: int
    latest_rank: int
    earliest_start: int
    latest_start: int
    min_end: int
    max_end: int
    skew_ns: int
    algo: str
    proto: str
    dtype: str
    count: int


def _stack_key(stack):
    return tuple(
        (frame.start, frame.end, frame.name)
        for frame in stack
        if not ALLREDUCE_RE.match(frame.name)
        and not frame.name.startswith("NcclGroup:")
    )


def _markers(conn):
    markers = []
    malformed = 0
    for row in conn.execute(MARK_SQL):
        match = ALLREDUCE_RE.match(row["text"])
        if not match:
            malformed += 1
            continue
        fields = match.groupdict()
        markers.append(
            Marker(
                start=row["start"],
                tid=row["tid"],
                comm=fields["comm"].lower(),
                seq=int(fields["seq"]),
                algo=fields["algo"].upper(),
                proto=fields["proto"].upper(),
                dtype=fields["dtype"],
                count=int(fields["count"]),
            )
        )
    return markers, malformed


def load_profile(db):
    """Associate markers and AllReduce launches by stack identity and order."""
    conn = open_db(db)
    require_kernel_table(conn, db)
    rank = get_rank(conn)
    if rank is None:
        conn.close()
        raise ValueError(f"{db}: profile has no RANK in DeviceEnvironment")
    epoch = read_epoch(conn)
    index = NvtxIndex(conn, rank)
    markers, malformed = _markers(conn)
    launches = conn.execute(LAUNCH_SQL).fetchall()

    marker_contexts = index.iter_stacks(
        (marker.start, marker.start) for marker in markers
    )
    launch_contexts = index.iter_stacks(
        (row["api_start"], row["api_end"]) for row in launches
    )
    by_context = defaultdict(list)
    for row, (_, _, stack) in zip(launches, launch_contexts):
        by_context[(row["tid"], _stack_key(stack))].append(row)

    cursors = defaultdict(int)
    matched = []
    unmatched = 0
    for marker, (_, _, stack) in zip(markers, marker_contexts):
        key = marker.tid, _stack_key(stack)
        rows = by_context.get(key, ())
        cursor = cursors[key]
        if cursor >= len(rows):
            unmatched += 1
            continue
        row = rows[cursor]
        cursors[key] = cursor + 1
        matched.append(
            AllReduceKernel(
                rank=rank,
                comm=marker.comm,
                seq=marker.seq,
                algo=marker.algo,
                proto=marker.proto,
                dtype=marker.dtype,
                count=marker.count,
                gpu_start=row["gpu_start"],
                gpu_end=row["gpu_end"],
                epoch=epoch,
            )
        )
    conn.close()
    return rank, epoch, matched, unmatched, malformed


def load_all(dbs, jobs):
    kernels = []

    def consume(result):
        rank, epoch, items, unmatched, malformed = result
        kernels.extend(items)
        print(
            f"  loaded rank {rank}: {len(items)} AllReduce kernels "
            f"({unmatched} unmatched, {malformed} malformed, "
            f"{'UTC' if epoch is not None else 'relative'})",
            file=sys.stderr,
        )

    if jobs <= 1 or len(dbs) == 1:
        for db in dbs:
            consume(load_profile(db))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(load_profile, db) for db in dbs]
            for future in as_completed(futures):
                consume(future.result())
    return kernels


def build_collectives(kernels, align_utc):
    """Build one max(start)-min(start) sample per complete (comm, seq)."""
    members = defaultdict(set)
    grouped = defaultdict(lambda: defaultdict(list))
    for item in kernels:
        members[item.comm].add(item.rank)
        grouped[(item.comm, item.seq)][item.rank].append(item)

    samples = []
    warnings = []
    for (comm, seq), by_rank in sorted(grouped.items()):
        expected = tuple(sorted(members[comm]))
        actual = tuple(sorted(by_rank))
        if actual != expected:
            missing = sorted(set(expected) - set(actual))
            warnings.append(f"{comm} seq={seq}: incomplete ranks; missing={missing}")
            continue
        duplicates = {
            rank: len(items) for rank, items in by_rank.items() if len(items) != 1
        }
        if duplicates:
            warnings.append(f"{comm} seq={seq}: duplicate rank kernels {duplicates}")
            continue
        items = [by_rank[rank][0] for rank in expected]
        metadata = {(item.algo, item.proto, item.dtype, item.count) for item in items}
        if len(metadata) != 1:
            warnings.append(f"{comm} seq={seq}: inconsistent marker metadata")
            continue
        aligned = []
        for item in items:
            base = item.epoch if align_utc and item.epoch is not None else 0
            aligned.append((item, base + item.gpu_start, base + item.gpu_end))
        earliest = min(aligned, key=lambda value: value[1])
        latest = max(aligned, key=lambda value: value[1])
        algo, proto, dtype, count = metadata.pop()
        samples.append(
            CollectiveSample(
                comm=comm,
                seq=seq,
                nranks=len(expected),
                member_ranks=expected,
                earliest_rank=earliest[0].rank,
                latest_rank=latest[0].rank,
                earliest_start=earliest[1],
                latest_start=latest[1],
                min_end=min(value[2] for value in aligned),
                max_end=max(value[2] for value in aligned),
                skew_ns=latest[1] - earliest[1],
                algo=algo,
                proto=proto,
                dtype=dtype,
                count=count,
            )
        )
    return samples, warnings


def percentile(values, q):
    import numpy as np

    return float(np.percentile(values, q))


def print_report(samples):
    grouped = defaultdict(list)
    widths = {}
    for sample in samples:
        grouped[sample.comm].append(sample.skew_ns)
        widths[sample.comm] = sample.nranks
    header = (
        f"{'commId':<20} {'nranks':>7} {'calls':>8} {'mean':>11} "
        f"{'p50':>11} {'p90':>11} {'p99':>11} {'max':>11}"
    )
    print(header)
    print("-" * len(header))
    for comm, values in sorted(grouped.items()):
        print(
            f"{comm:<20} {widths[comm]:>7} {len(values):>8} "
            f"{human_ns(sum(values) / len(values)):>11} "
            f"{human_ns(percentile(values, 50)):>11} "
            f"{human_ns(percentile(values, 90)):>11} "
            f"{human_ns(percentile(values, 99)):>11} {human_ns(max(values)):>11}"
        )
    values = [sample.skew_ns for sample in samples]
    print(
        f"\nAll collectives: {len(values)}, mean={human_ns(sum(values) / len(values))}, "
        f"p50={human_ns(percentile(values, 50))}, "
        f"p99={human_ns(percentile(values, 99))}, max={human_ns(max(values))}"
    )


def render_histogram(samples, out, bins):
    if bins is not None and bins < 1:
        raise ValueError("histogram bins must be positive")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import AutoMinorLocator, MaxNLocator

    values = np.asarray([sample.skew_ns for sample in samples]) / 1e6
    percentile_hi = max(float(np.percentile(values, 99.5)), float(np.finfo(float).eps))
    cell_ms = 0.25
    display_hi = max(cell_ms, np.ceil(percentile_hi * 1.25 / cell_ms) * cell_ms)
    overflow = int(np.count_nonzero(values > display_hi))
    edges = (
        np.arange(0, display_hi + cell_ms / 2, cell_ms)
        if bins is None
        else np.linspace(0, display_hi, bins + 1)
    )
    style = {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "hatch.linewidth": 0.45,
        "savefig.facecolor": "white",
    }
    with plt.rc_context(style):
        fig, ax = plt.subplots(figsize=(6.5, 3.35), constrained_layout=True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="in", length=3, width=0.7)
        ax.set_axisbelow(True)
        ax.grid(
            axis="both",
            which="major",
            color="#D8D8D8",
            linewidth=0.5,
            alpha=0.75,
        )
        ax.grid(
            axis="x",
            which="minor",
            color="#E8E8E8",
            linewidth=0.4,
            alpha=0.65,
        )
        ax.hist(
            values,
            bins=edges,
            color="#A9C7E8",
            edgecolor="#4F81BD",
            alpha=0.72,
            hatch="////",
            linewidth=0.55,
            zorder=2,
        )
        ax.hist(
            values,
            bins=edges,
            histtype="step",
            color="#244A7C",
            linewidth=1.25,
            zorder=3,
        )
        ax.set_xlim(0, display_hi)
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7, min_n_ticks=5))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.set_xlabel(r"GPU kernel start-time skew, $t_{max}-t_{min}$ (ms)")
        ax.set_ylabel("AllReduce calls (log scale)")
        ax.set_title(
            f"AllReduce kernel-start skew distribution  [n={len(values):,}]",
            loc="left",
            fontweight="bold",
            pad=5,
        )
        summary = (
            f"p50 {np.percentile(values, 50):.2f} ms   "
            f"p99 {np.percentile(values, 99):.2f} ms\n"
            f"xmax {display_hi:.2f} ms; overflow {overflow:,}"
        )
        ax.text(
            0.985,
            0.955,
            summary,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#202020",
            linespacing=1.35,
        )
    path = Path(out)
    path = path if path.suffix.lower() == ".png" else path.with_suffix(".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return str(path)


def write_csv(samples, out):
    path = Path(out)
    path = path if path.suffix.lower() == ".csv" else path.with_suffix(".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "comm_id",
                "seq",
                "nranks",
                "member_ranks",
                "earliest_rank",
                "latest_rank",
                "earliest_start_ns",
                "latest_start_ns",
                "min_end_ns",
                "max_end_ns",
                "skew_ns",
                "algo",
                "proto",
                "dtype",
                "count",
            ]
        )
        for item in samples:
            writer.writerow(
                [
                    item.comm,
                    item.seq,
                    item.nranks,
                    ";".join(map(str, item.member_ranks)),
                    item.earliest_rank,
                    item.latest_rank,
                    item.earliest_start,
                    item.latest_start,
                    item.min_end,
                    item.max_end,
                    item.skew_ns,
                    item.algo,
                    item.proto,
                    item.dtype,
                    item.count,
                ]
            )
    return str(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Match AllReduce NVTX markers to GPU kernels and plot start skew."
    )
    parser.add_argument("db", nargs="+", help="per-rank .sqlite profiles")
    parser.add_argument("--png", default="gpu_allreduce_skew.png", metavar="OUT")
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument("--csv", default=None, metavar="OUT")
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=None,
        metavar="N",
        help="override the default 0.25 ms histogram-bin width with N bins",
    )
    parser.add_argument("--no-align", action="store_true")
    parser.add_argument(
        "--jobs", type=int, default=min(8, os.cpu_count() or 1), metavar="N"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    kernels = load_all(args.db, args.jobs)
    if not kernels:
        sys.exit("no AllReduce markers with matching GPU kernel launches found")
    align = not args.no_align and all(item.epoch is not None for item in kernels)
    if not args.no_align and not align and len(args.db) > 1:
        print(
            "warning: UTC epoch missing; profile-relative skew may be invalid",
            file=sys.stderr,
        )
    samples, warnings = build_collectives(kernels, align)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if not samples:
        sys.exit("no complete cross-rank AllReduce collectives found")
    print(f"Timestamp alignment: {'session UTC' if align else 'profile-relative'}\n")
    print_report(samples)
    if args.csv:
        print(f"\nWrote collective CSV -> {write_csv(samples, args.csv)}")
    if not args.no_png:
        print(
            f"Wrote skew histogram -> {render_histogram(samples, args.png, args.hist_bins)}"
        )
