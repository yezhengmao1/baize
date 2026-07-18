"""Match P2P NVTX markers to real GPU kernels and histogram start-time skew."""

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

P2P_RE = re.compile(
    r"^P2p:commId=(?P<comm>0x[0-9a-fA-F]+):rank=(?P<rank>\d+)"
    r":peer=(?P<peer>\d+):seq=(?P<seq>\d+)(?:.*?:func=(?P<func>Send|Recv))?",
    re.IGNORECASE,
)
P2P_MARK_SQL = """
SELECT n.start, n.globalTid tid, COALESCE(n.text, s.value) text
FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId
WHERE COALESCE(n.text, s.value) LIKE 'P2p:commId=%' ORDER BY n.start
"""
LAUNCH_SQL = """
SELECT r.start api_start, r.end api_end, r.globalTid tid,
       k.start gpu_start, k.end gpu_end, sk.value kernel_name
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=k.correlationId
JOIN StringIds sa ON sa.id=r.nameId JOIN StringIds sk ON sk.id=k.shortName
WHERE sa.value LIKE '%LaunchKernel%' ORDER BY r.start
"""


@dataclass(frozen=True)
class Marker:
    start: int
    tid: int
    comm: str
    rank: int
    peer: int
    seq: int


@dataclass(frozen=True)
class P2PKernel:
    comm: str
    seq: int
    rank: int
    peer: int
    gpu_start: int
    gpu_end: int
    epoch: int | None


@dataclass(frozen=True)
class PairSample:
    comm: str
    seq: int
    rank_a: int
    rank_b: int
    start_a: int
    start_b: int
    end_a: int
    end_b: int
    skew_ns: int


def _stack_key(stack):
    # Some exporters represent an NVTX mark as a zero-length range. Exclude the
    # P2P marker itself so it cannot make the marker and launch stacks differ.
    return tuple(
        (frame.start, frame.end, frame.name)
        for frame in stack
        if not P2P_RE.match(frame.name) and not frame.name.startswith("NcclGroup:")
    )


def _markers(conn):
    result = []
    malformed = 0
    for row in conn.execute(P2P_MARK_SQL):
        match = P2P_RE.match(row["text"])
        if not match:
            malformed += 1
            continue
        item = match.groupdict()
        result.append(
            Marker(
                row["start"],
                row["tid"],
                item["comm"].lower(),
                int(item["rank"]),
                int(item["peer"]),
                int(item["seq"]),
            )
        )
    return result, malformed


def load_profile(db):
    """Associate marker and launch by exact enclosing-stack identity and order."""
    conn = open_db(db)
    require_kernel_table(conn, db)
    rank, epoch = get_rank(conn), read_epoch(conn)
    index = NvtxIndex(conn, rank)
    markers, malformed = _markers(conn)
    launches = conn.execute(LAUNCH_SQL).fetchall()
    marker_contexts = index.iter_stacks((m.start, m.start) for m in markers)
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
        rows, cursor = by_context[key], cursors[key]
        if cursor == len(rows):
            unmatched += 1
            cursors[key] = cursor
            continue
        row = rows[cursor]
        cursors[key] = cursor + 1
        matched.append(
            P2PKernel(
                marker.comm,
                marker.seq,
                marker.rank,
                marker.peer,
                row["gpu_start"],
                row["gpu_end"],
                epoch,
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
            f"  loaded rank {rank}: {len(items)} P2P kernels "
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


def build_pairs(kernels, align_utc):
    """Pair by commId + seq + unordered endpoint ranks."""
    grouped = defaultdict(lambda: defaultdict(list))
    for item in kernels:
        a, b = sorted((item.rank, item.peer))
        grouped[(item.comm, item.seq, a, b)][item.rank].append(item)
    samples, warnings = [], []
    for (comm, seq, a, b), ranks in sorted(grouped.items()):
        left = sorted(ranks.get(a, []), key=lambda item: item.gpu_start)
        right = sorted(ranks.get(b, []), key=lambda item: item.gpu_start)
        if len(left) != len(right):
            warnings.append(
                f"{comm} seq={seq} r{a}<->r{b}: endpoint counts "
                f"{len(left)} vs {len(right)}; paired common prefix"
            )
        for x, y in zip(left, right):
            bx = x.epoch if align_utc and x.epoch is not None else 0
            by = y.epoch if align_utc and y.epoch is not None else 0
            sx, sy = bx + x.gpu_start, by + y.gpu_start
            samples.append(
                PairSample(
                    comm,
                    seq,
                    a,
                    b,
                    sx,
                    sy,
                    bx + x.gpu_end,
                    by + y.gpu_end,
                    abs(sx - sy),
                )
            )
    return samples, warnings


def percentile(values, q):
    import numpy as np

    return float(np.percentile(values, q))


def print_report(samples):
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample.comm].append(sample.skew_ns)
    header = (
        f"{'commId':<20} {'pairs':>8} {'mean':>11} {'p50':>11} "
        f"{'p90':>11} {'p99':>11} {'max':>11}"
    )
    print(header)
    print("-" * len(header))
    for comm, values in sorted(grouped.items()):
        print(
            f"{comm:<20} {len(values):>8} {human_ns(sum(values) / len(values)):>11} "
            f"{human_ns(percentile(values, 50)):>11} "
            f"{human_ns(percentile(values, 90)):>11} "
            f"{human_ns(percentile(values, 99)):>11} {human_ns(max(values)):>11}"
        )
    values = [sample.skew_ns for sample in samples]
    print(
        f"\nAll pairs: {len(values)}, mean={human_ns(sum(values) / len(values))}, "
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
    if bins is None:
        edges = np.arange(0, display_hi + cell_ms / 2, cell_ms)
    else:
        edges = np.linspace(0, display_hi, bins + 1)
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
        ax.set_xlabel(r"GPU kernel start-time skew, $|t_a-t_b|$ (ms)")
        ax.set_ylabel("Communication pairs (log scale)")
        ax.set_title(
            f"P2P kernel-start skew distribution  [n={len(values):,}]",
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
                "rank_a",
                "rank_b",
                "gpu_start_a_ns",
                "gpu_start_b_ns",
                "gpu_end_a_ns",
                "gpu_end_b_ns",
                "skew_ns",
            ]
        )
        for item in samples:
            writer.writerow(
                [
                    item.comm,
                    item.seq,
                    item.rank_a,
                    item.rank_b,
                    item.start_a,
                    item.start_b,
                    item.end_a,
                    item.end_b,
                    item.skew_ns,
                ]
            )
    return str(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Match P2p:commId markers to GPU kernels and plot start skew."
    )
    parser.add_argument("db", nargs="+", help="per-rank .sqlite profiles")
    parser.add_argument("--png", default="gpu_p2p_skew.png", metavar="OUT")
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
        sys.exit("no P2p:commId markers with matching GPU kernel launches found")
    align = not args.no_align and all(item.epoch is not None for item in kernels)
    if not args.no_align and not align and len(args.db) > 1:
        print(
            "warning: UTC epoch missing; profile-relative skew may be invalid",
            file=sys.stderr,
        )
    samples, warnings = build_pairs(kernels, align)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if not samples:
        sys.exit("no complete P2P communication pairs found")
    print(f"Timestamp alignment: {'session UTC' if align else 'profile-relative'}\n")
    print_report(samples)
    if args.csv:
        print(f"\nWrote pair CSV -> {write_csv(samples, args.csv)}")
    if not args.no_png:
        print(
            f"Wrote skew histogram -> {render_histogram(samples, args.png, args.hist_bins)}"
        )
