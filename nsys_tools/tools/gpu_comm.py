"""
Cross-rank communication skew report: per-(NVTX scope, op) collectives + P2P +
DeepEP, reporting per-communication-group call counts and worst-wait distribution.
Author: yezhengmaolove@gmail.com
"""

import argparse
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

from .flamegraph import compute_step_windows, detect_steps
from ..utils.comm import CommEvent, load_comm_in_window
from ..utils.nvtx import NvtxIndex
from ..utils.common import get_rank, human_ns, open_db, require_kernel_table

matplotlib.use("Agg")

LOCAL_WORLD = 8
MIN_HEATMAP_VOL = 1 << 30


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


# =============================================================================
# Per-rank load (windowed)
# =============================================================================

TRACE_START_SQL = "SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL"
TRACE_END_SQL = "SELECT MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL"


def _window(conn, idx: NvtxIndex, needle: str, skip: int) -> tuple[int, int, int]:
    """
    return (window_start, window_end, n_steps) for one profile
    """
    steps = detect_steps(idx, needle)
    t_min = conn.execute(TRACE_START_SQL).fetchone()[0]
    if len(steps) <= skip:
        t_max = conn.execute(TRACE_END_SQL).fetchone()[0]
        return t_min, t_max, len(steps)
    _, window_start, window_end = compute_step_windows(steps, t_min, skip)
    return window_start, window_end, len(steps)


def load_rank(
    db: str, needle: str, skip: int, dtype_bytes: int = 2
) -> tuple[int | None, list[CommEvent], int, int]:
    """
    Load comm events for one rank within the post-warmup window.
    Return (rank, events, window_duration, n_steps).
    """
    conn = open_db(db)
    require_kernel_table(conn, db)
    rank = get_rank(conn)
    idx = NvtxIndex(conn, rank)
    window_start, window_end, n_steps = _window(conn, idx, needle, skip)
    events = load_comm_in_window(conn, idx, window_start, window_end, dtype_bytes)
    conn.close()
    return rank, events, window_end - window_start, n_steps


def load_all(
    dbs: list[str], needle: str, skip: int, dtype_bytes: int, jobs: int
) -> tuple[list[tuple[int | None, list[CommEvent]]], list[int], int]:
    """Load every rank, optionally across a process pool (ranks are independent
    files, so loading is embarrassingly parallel). Returns (records, durs, nsteps);
    order is not significant — the report only aggregates."""
    records: list[tuple[int | None, list[CommEvent]]] = []
    durs: list[int] = []
    nsteps = 0

    def consume(rank, events, wdur, nstep):
        nonlocal nsteps
        records.append((rank, events))
        durs.append(wdur)
        nsteps = max(nsteps, nstep)
        print(f"  loaded rank {rank}: {len(events)} comm kernels", file=sys.stderr)

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
# Report
# =============================================================================


class ReportData(NamedTuple):
    by_scope: dict
    p2p: dict
    ep: dict


def _accumulate(records: list[tuple[int, list[CommEvent]]]):
    """Fold per-rank events into cross-rank accumulators."""

    # collectives, keyed by (comm, op, seq) — the comm handle is the PG identity, so
    # durs collects this one collective's per-rank gpu_dur across exactly its members.
    # durs: {rank: gpu_dur}; bytes: per-rank msg size; scope: issuing NVTX scope.
    inst: dict[tuple, dict] = defaultdict(lambda: {"durs": {}, "bytes": 0, "scope": ""})

    # P2P send/recv, keyed by scope -> rank; slot = [bytes, gpu_dur, calls].
    # Peer-based, no group, so it stays per-rank.
    p2p: dict[str, dict] = defaultdict(
        lambda: defaultdict(lambda: {"Send": [0, 0, 0], "Recv": [0, 0, 0]})
    )

    # DeepEP, keyed by scope -> rank; slot = [bytes, gpu_dur, calls]. Stored per rank;
    # the EP communication group is applied later in _print_ep (--ep N: consecutive
    # ranks rank // N, e.g. ep=8 -> ranks 0..7 are one group) to get within-group skew.
    ep: dict[str, dict] = defaultdict(
        lambda: defaultdict(lambda: {"Dispatch": [0, 0, 0], "Combine": [0, 0, 0]})
    )

    for rank, events in records:
        for e in events:
            if e.comm and e.seq is not None:
                rec = inst[(e.comm, e.op, e.seq)]
                rec["durs"][rank] = e.gpu_dur
                rec["bytes"] = e.bytes
                rec["scope"] = e.scope
            elif e.op in ("Send", "Recv"):
                slot = p2p[e.scope][rank][e.op]
                slot[0] += e.bytes
                slot[1] += e.gpu_dur
                slot[2] += 1
            elif e.op in ("Dispatch", "Combine"):
                slot = ep[e.scope][rank][e.op]
                slot[0] += e.bytes
                slot[1] += e.gpu_dur
                slot[2] += 1
    return inst, p2p, ep


def _aggregate_by_scope(inst: dict) -> dict:
    """Roll cross-rank-aligned collective instances up to per-(scope, op, width),
    where width = len(durs) = the collective's true group size. Within each bucket
    every comm handle is one PG (group); groups[comm][rank] = [Σtime, calls] keeps
    each PG separate so the report can summarize per-group skew across the replicated
    groups. bytes / wait_by_rank are kept for the heatmap."""
    by_scope: dict[str, dict] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "groups": defaultdict(lambda: defaultdict(lambda: [0.0, 0])),
                "bytes": 0,
                "wait_by_rank": defaultdict(float),
            }
        )
    )
    for (comm, op, _seq), d in inst.items():
        durs = d["durs"]
        if not durs:
            continue
        floor = min(durs.values())
        width = len(durs)
        agg = by_scope[d["scope"]][(op, width)]
        agg["bytes"] += d["bytes"]
        grp = agg["groups"][comm]
        for r, t in durs.items():
            grp[r][0] += t
            grp[r][1] += 1
            agg["wait_by_rank"][r] += t - floor
    return by_scope


def _print_collectives(by_scope: dict, n: int) -> None:
    """Collectives by NVTX scope, aligned by (comm, op, seq). Within each
    (scope, op, width) every comm handle is one PG; per PG we take its worst-rank
    wait (max-min of per-rank time) and call count, and the row reports the
    distribution across the replicated PGs — worst time as min/mean/max, calls as
    min/max."""
    if not by_scope:
        return
    sw = max(len("scope"), min(40, max(len(s) for s in by_scope)))
    header = (
        f"{'scope':<{sw}} {'nranks':>6} {'op':<13} {'calls min':>9} "
        f"{'calls max':>9} {'worst min':>11} {'worst mean':>11} {'worst max':>11}"
    )
    print(f"=== Collectives by NVTX scope, {n} ranks sampled ===")
    print(header)
    print("-" * len(header))
    vol = lambda kv: kv[1]["bytes"] * kv[0][1]  # bytes * width = aggregate traffic
    order = sorted(by_scope, key=lambda s: -sum(vol(kv) for kv in by_scope[s].items()))
    for scope in order:
        first = True
        for (op, width), agg in sorted(
            by_scope[scope].items(), key=lambda kv: -vol(kv)
        ):
            worsts: list[float] = []  # per-PG worst wait = max-min of rank times
            callses: list[int] = []  # per-PG call count (per rank in the group)
            for per_rank in agg["groups"].values():
                times = [tc[0] for tc in per_rank.values()]
                calls = [tc[1] for tc in per_rank.values()]
                worsts.append(max(times) - min(times))
                callses.append(round(sum(calls) / len(calls)))
            lab = scope[:sw] if first else ""
            first = False
            print(
                f"{lab:<{sw}} {width:>6} {op:<13} {min(callses):>9} "
                f"{max(callses):>9} {human_ns(min(worsts)):>11} "
                f"{human_ns(sum(worsts) / len(worsts)):>11} {human_ns(max(worsts)):>11}"
            )
    print()


def _print_p2p(p2p: dict) -> None:
    """P2P send/recv per issuing scope. No comm/seq and peer-based → no clean group,
    so only group-independent quantities: Σvol/rank, calls/rank, per-rank time spread."""
    if not p2p:
        return
    nr = len({r for br in p2p.values() for r in br})
    sw = max(len("scope"), min(40, max(len(s) for s in p2p)))
    header = (
        f"{'scope':<{sw}} {'op':<5} {'calls':>6} {'Σvol/rank':>10} "
        f"{'time floor':>12} {'time max':>12}"
    )
    print(
        f"=== P2P send/recv by NVTX scope, {nr} ranks (no group; per-rank totals) ==="
    )
    print(header)
    print("-" * len(header))
    order = sorted(
        p2p, key=lambda s: -sum(d["Send"][0] + d["Recv"][0] for d in p2p[s].values())
    )
    for scope in order:
        br = p2p[scope]
        first = True
        for op in ("Send", "Recv"):
            times = [d[op][1] for d in br.values()]
            if not any(times):
                continue
            n = len(br)
            vol = sum(d[op][0] for d in br.values()) / n
            calls = round(sum(d[op][2] for d in br.values()) / n)
            lab = scope[:sw] if first else ""
            first = False
            print(
                f"{lab:<{sw}} {op:<5} {calls:>6} {human_bytes(vol):>10} "
                f"{human_ns(min(times)):>12} {human_ns(max(times)):>12}"
            )
    print()


def _print_ep(ep: dict, ep_group: int) -> None:
    """DeepEP dispatch/combine per scope. Ranks are partitioned into EP groups of
    ep_group consecutive ranks (rank // ep_group, e.g. ep=8 -> ranks 0..7 = one
    group). For each group we take its worst-rank wait (max-min of per-rank time)
    and its call count; the row reports the distribution of those across all groups
    — worst time as min/mean/max, calls as min/max."""
    if not ep:
        return
    sw = max(len("scope"), min(40, max(len(s) for s in ep)))
    header = (
        f"{'scope':<{sw}} {'nranks':>6} {'phase':<8} {'calls min':>9} "
        f"{'calls max':>9} {'worst min':>11} {'worst mean':>11} {'worst max':>11}"
    )
    print(f"=== EP all-to-all (DeepEP) by NVTX scope, EP group = {ep_group} ranks ===")
    print(header)
    print("-" * len(header))
    order = sorted(
        ep,
        key=lambda s: -sum(d["Dispatch"][0] + d["Combine"][0] for d in ep[s].values()),
    )
    for scope in order:
        br = ep[scope]
        first = True
        for phase in ("Dispatch", "Combine"):
            if not any(d[phase][1] for d in br.values()):
                continue
            # group ranks into EP groups; per group keep (time, calls) of its ranks
            groups: dict[int, list] = defaultdict(list)
            for rank, d in br.items():
                groups[rank // ep_group].append((d[phase][1], d[phase][2]))
            worsts: list[float] = []  # per-group worst wait = max-min of rank times
            callses: list[int] = []  # per-group call count (per rank in the group)
            for members in groups.values():
                times = [t for t, _ in members]
                worsts.append(max(times) - min(times))
                callses.append(round(sum(c for _, c in members) / len(members)))
            if not worsts:
                continue
            lab = scope[:sw] if first else ""
            first = False
            print(
                f"{lab:<{sw}} {ep_group:>6} {phase:<8} {min(callses):>9} "
                f"{max(callses):>9} {human_ns(min(worsts)):>11} "
                f"{human_ns(sum(worsts) / len(worsts)):>11} {human_ns(max(worsts)):>11}"
            )
    print()


def report(records: list[tuple[int, list[CommEvent]]], ep_group: int) -> ReportData:
    inst, p2p, ep = _accumulate(records)
    by_scope = _aggregate_by_scope(inst)
    _print_collectives(by_scope, len(records))
    _print_ep(ep, ep_group)
    _print_p2p(p2p)
    return ReportData(by_scope, p2p, ep)


def _by_node(d: dict, node_size: int) -> dict:
    """Average a {rank: value} column over the ranks of each node."""
    acc: dict = defaultdict(lambda: [0.0, 0])
    for rank, v in d.items():
        a = acc[rank // node_size]
        a[0] += v
        a[1] += 1
    return {node: s / n for node, (s, n) in acc.items()}


def build_heatmap_columns(by_scope, p2p, ep, node_size: int = 0):
    """Prepare heatmap columns: list of (label, {row: value}, kind).

    One wait column per high-volume (scope, op) — the scope grouping already
    collapses the replicated HSDP shards. P2P send/recv and EP are appended as
    time columns.

    node_size > 0 aggregates rows by node (rank // node_size), averaging each
    column over the node's ranks — fewer, cleaner rows that surface slow nodes.
    """
    cols = []  # (label, {row: value}, kind): 'wait' (low=slow) or 'time' (high=slow)
    pairs = (
        (scope, op, agg)
        for scope, ops in by_scope.items()
        for (op, _w), agg in ops.items()
    )
    for scope, op, agg in sorted(pairs, key=lambda x: -x[2]["bytes"]):
        if agg["bytes"] >= MIN_HEATMAP_VOL:
            cols.append((f"{scope[:12]}:{op[:5]}", dict(agg["wait_by_rank"]), "wait"))

    # P2P / EP are nested {scope: {rank: data}}; flatten the per-rank time across
    # scopes into one column per op (the heatmap is about per-rank slowness).
    def _flatten(by_scope_rank, op):
        out: dict = defaultdict(float)
        for by_rank in by_scope_rank.values():
            for r, d in by_rank.items():
                out[r] += d[op][1]
        return dict(out)

    for op in ("Send", "Recv"):
        col = _flatten(p2p, op)
        if any(col.values()):
            cols.append((f"P2P:{op}", col, "time"))
    for op in ("Dispatch", "Combine"):
        col = _flatten(ep, op)
        if any(col.values()):
            cols.append((f"EP:{op[:5]}", col, "time"))
    if node_size > 0:
        cols = [(lab, _by_node(d, node_size), kind) for lab, d, kind in cols]
    return cols


def render_heatmap(
    cols: list[tuple[str, dict, str]],
    out: str,
    title: str = "",
    row_prefix: str = "r",
    sort: str = "slowness",
) -> str:
    """Draw the ranks × comms skew heatmap and return the written PNG path.

    cols: list of (label, {row: value}, kind). kind = 'wait' (low value = slow
    straggler) or 'time' (high value = slow). Values are normalized per column to
    [0, 1] with bright = slowest row. sort: "slowness" (default, stragglers on
    top) or "id" (rows in ascending node/rank order).
    """
    ranks = sorted({r for _, d, _ in cols for r in d})
    ridx = {r: i for i, r in enumerate(ranks)}
    M = np.full((len(ranks), len(cols)), np.nan)
    for j, (_label, d, kind) in enumerate(cols):
        vals = list(d.values())
        lo, hi = min(vals), max(vals)
        for r, v in d.items():
            if hi == lo:
                s = 0.5
            elif kind == "wait":  # low wait = straggler -> bright
                s = (hi - v) / (hi - lo)
            else:  # high time = slow -> bright
                s = (v - lo) / (hi - lo)
            M[ridx[r], j] = s

    if sort == "id":  # ascending node/rank order (ranks already sorted)
        order = list(range(len(ranks)))
    else:  # stragglers on top
        order = sorted(range(len(ranks)), key=lambda i: -np.nanmean(M[i]))
    M = M[order]
    rlabels = [f"{row_prefix}{ranks[i]}" for i in order]

    fig_w = max(7, 0.5 * len(cols) + 3)
    fig_h = max(4, 0.13 * len(ranks) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#dddddd")
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[0] for c in cols], rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(rlabels)))
    ax.set_yticklabels(rlabels, fontsize=5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label(
        "slowness in this comm (1 = straggler / others wait for it)", fontsize=8
    )
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    png = Path(out)
    png = png if png.suffix.lower() == ".png" else Path(str(out) + ".png")
    fig.savefig(png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(png)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-rank communication skew report (per-group calls + worst-wait)"
    )
    p.add_argument("db", nargs="+", help="One or more .sqlite profiles")
    p.add_argument("--step-nvtx", default="Optimizer.step", metavar="SUBSTR")
    p.add_argument("--skip-steps", type=int, default=1, metavar="N")
    p.add_argument(
        "--p2p-dtype-bytes",
        type=int,
        default=2,
        metavar="N",
        help="bytes/element for marker-less P2P + DeepEP volume; "
        "default 2 (bf16), 1 for fp8, 4 for fp32",
    )
    p.add_argument(
        "--ep",
        type=int,
        default=LOCAL_WORLD,
        metavar="N",
        help=f"DeepEP all-to-all group size (consecutive ranks rank//N form one "
        f"group); EP skew is computed within groups. Default {LOCAL_WORLD} (intranode)",
    )
    p.add_argument(
        "--heatmap",
        default=None,
        metavar="OUT",
        help="also write a per-rank × (PG,op) skew heatmap PNG",
    )
    p.add_argument(
        "--by-node",
        action="store_true",
        help=f"aggregate heatmap rows by node ({LOCAL_WORLD} ranks each), averaging "
        "— fewer rows, surfaces slow nodes",
    )
    p.add_argument(
        "--sort",
        choices=("slowness", "id"),
        default="slowness",
        help="heatmap row order: 'slowness' (stragglers on top) or 'id' "
        "(ascending node/rank order)",
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

    records: list[tuple[int, list[CommEvent]]]
    durs: list[int]
    nsteps: int

    records, durs, nsteps = load_all(
        args.db,
        args.step_nvtx,
        args.skip_steps,
        args.p2p_dtype_bytes,
        args.jobs,
    )

    wmin, wmax = min(durs), max(durs)
    print(
        f"Window: '{args.step_nvtx}' markers={nsteps}, skipping first "
        f"{args.skip_steps} (warmup); post-warmup wall ≈ "
        f"{human_ns(wmin)}"
        + (f"–{human_ns(wmax)}" if wmax - wmin > 1e6 else "")
        + f" over {len(records)} ranks\n"
    )
    data = report(records, args.ep)

    if args.heatmap:
        node_size = LOCAL_WORLD if args.by_node else 0
        cols = build_heatmap_columns(
            data.by_scope,
            data.p2p,
            data.ep,
            node_size,
        )
        if not cols:
            print("(heatmap: no high-volume columns)")
        else:
            unit = "nodes" if args.by_node else "ranks"
            nrow = len({k for _, d, _ in cols for k in d})
            title = (
                f"Comm skew heatmap — {nrow} {unit} × {len(cols)} comms\n"
                "bright = straggler; rows sorted by mean slowness"
            )
            out = render_heatmap(
                cols,
                args.heatmap,
                title,
                row_prefix="node " if args.by_node else "r",
                sort=args.sort,
            )
            print(f"\nWrote comm skew heatmap: {out}")
