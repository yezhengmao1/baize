"""
Communication extraction: per-kernel comm records from one profile.

Covers two comm backends:
  - NCCL collectives / P2P  (ncclDevKernel, joined to nccl:<op> ranges + comm= markers)
  - DeepEP MoE all-to-all   (deep_ep::{intra,inter}node::{dispatch,combine,notify,…}
                             kernels, joined to FusedDispatch/FusedCombine ranges)

Author: yezhengmaolove@gmail.com
"""

import bisect
import re
import sqlite3
from typing import NamedTuple

from .nvtx import NvtxIndex

# Strip per-instance tails so scope names aggregate; drop comm-plumbing frames so
# the semantic enclosing scope (e.g. "mlp forward", "_LinearBackward") is kept.
_NVTX_TAIL = re.compile(
    r"\s*,\s*(op_id|seq|sizes|input_op_ids|input_shapes|dtype|count)\b.*$"
)
_PLUMBING = ("nccl", "c10d::", "record_param_comms", "NcclGroup")


def _scope_from_stack(stack) -> str:
    """Innermost non-plumbing NVTX frame = the model-level scope that issued the
    comm (e.g. '_LinearBackward'). stack is outermost-first. If every frame is
    comm plumbing (e.g. a bare c10d all-reduce with no model scope around it), fall
    back to the outermost frame — that call is still the most meaningful label."""
    sem = None
    for e in stack:
        n = _NVTX_TAIL.sub("", e.name)
        if not n.startswith(_PLUMBING):
            sem = n
    if sem is not None:
        return sem
    return _NVTX_TAIL.sub("", stack[0].name) if stack else "<none>"


# =============================================================================
# SQL
# =============================================================================

# ncclDevKernel launches with CPU launch time, thread, GPU duration, name.
NCCL_KERNEL_SQL = """
SELECT r.start AS api_start, r.end AS api_end, r.globalTid AS tid,
       k.start AS gpu_start, k.end AS gpu_end, k.end - k.start AS gpu_dur,
       s.value AS kernel_name
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId
JOIN StringIds s ON s.id = k.shortName
WHERE s.value LIKE 'ncclDevKernel%'
  AND k.start >= ? AND k.start <= ?
ORDER BY r.start
"""

# All nccl:<op> semantic ranges (the per-op enclosing scope), with thread + text.
NCCL_RANGE_SQL = """
SELECT start, end, globalTid AS tid, text
FROM NVTX_EVENTS
WHERE text LIKE 'nccl:%' AND end > start
ORDER BY start
"""

# The launch-time detail markers (PG handle / algo / dtype / count).
COMM_MARK_SQL = """
SELECT start, globalTid AS tid, text
FROM NVTX_EVENTS
WHERE text LIKE '%:comm=0x%'
ORDER BY start
"""

# Per-rank position within each NCCL group (max+1 across ranks = nranks).
NCCL_GROUP_SQL = "SELECT text FROM NVTX_EVENTS WHERE text LIKE 'NcclGroup:rank=%'"

# DeepEP MoE all-to-all kernels. Filter by the deep_ep:: namespace on the
# demangled name (robust against the unqualified short names "dispatch"/"combine");
# shortName classifies the phase, full name carries intranode/internode topology.
# No comm= marker and no nccl: range — volume comes from the enclosing
# FusedDispatch/FusedCombine NVTX range's sizes (attributed to the data-moving
# dispatch/combine kernel only; the notify/layout helpers carry 0 bytes).
DEEPEP_KERNEL_SQL = """
SELECT r.start AS api_start, r.end AS api_end, r.globalTid AS tid,
       k.start AS gpu_start, k.end - k.start AS gpu_dur,
       sn.value AS kernel_name, dm.value AS full_name
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId = k.correlationId
JOIN StringIds dm ON dm.id = k.demangledName
JOIN StringIds sn ON sn.id = k.shortName
WHERE dm.value LIKE '%deep_ep::%'
  AND k.start >= ? AND k.start <= ?
ORDER BY r.start
"""


# =============================================================================
# Parsing
# =============================================================================

COMM_MARK_RE = re.compile(
    r"^(?P<op>\w+):comm=(?P<comm>0x[0-9a-fA-F]+):seq=(?P<seq>\d+)"
    r":algo=(?P<algo>\w+):proto=(?P<proto>\w+):dtype=(?P<dtype>\w+):count=(?P<count>\d+)"
)

# nccl:<base> range -> canonical op. send/recv handled separately (carry a peer).
_OP_BY_BASE = {
    "_all_gather_base": "AllGather",
    "ALLGATHER_coalesced": "AllGather",
    "_reduce_scatter_base": "ReduceScatter",
    "REDUCESCATTER_coalesced": "ReduceScatter",
    "all_reduce": "AllReduce",
    "allreduce_coalesced": "AllReduce",
    "all_reduce_barrier": "AllReduce",
    "_broadcast_oop": "Broadcast",
    "broadcast": "Broadcast",
}

DTYPE_BYTES = {
    "ncclInt8": 1,
    "ncclUint8": 1,
    "ncclChar": 1,
    "ncclFloat16": 2,
    "ncclHalf": 2,
    "ncclBfloat16": 2,
    "ncclInt32": 4,
    "ncclUint32": 4,
    "ncclFloat32": 4,
    "ncclFloat": 4,
    "ncclInt64": 8,
    "ncclUint64": 8,
    "ncclFloat64": 8,
    "ncclDouble": 8,
}

_SIZES_RE = re.compile(r"sizes = \[(.*?)\]\]")
_NUMS_RE = re.compile(r"\[([0-9, ]*)\]")
_FIRST_TENSOR_RE = re.compile(r"sizes = \[\[([0-9, ]*)\]")
# innermost Fused{Dispatch,Combine}[Backward] frame = the DeepEP op range
_DEEPEP_RANGE_RE = re.compile(r"^Fused(?:Dispatch|Combine)")
_SEND_RE = re.compile(r"^nccl:send (\d+)->(\d+)")
_RECV_RE = re.compile(r"^nccl:recv (\d+)<-(\d+)")
_BASE_RE = re.compile(r"^nccl:([A-Za-z_]+)")


def _range_elems(text: str) -> int:
    """Total element count from a range's sizes = [[..],[..]] field (Σ of products)."""
    m = _SIZES_RE.search(text)
    if not m:
        return 0
    total = 0
    for grp in _NUMS_RE.findall("[" + m.group(1) + "]]"):
        dims = [int(x) for x in grp.split(",") if x.strip()]
        if dims:
            p = 1
            for d in dims:
                p *= d
            total += p
    return total


def _first_tensor_elems(text: str) -> int:
    """Product of the first sizes = [[d0, d1, …], …] tensor (the token payload)."""
    m = _FIRST_TENSOR_RE.search(text)
    if not m:
        return 0
    dims = [int(x) for x in m.group(1).split(",") if x.strip()]
    p = 1
    for d in dims:
        p *= d
    return p if dims else 0


def _deepep_frame(stack):
    """Innermost enclosing FusedDispatch/FusedCombine NVTX frame, if present."""
    frame = None
    for e in stack:
        if _DEEPEP_RANGE_RE.match(e.name):
            frame = e
    return frame


def _deepep_volume(stack, dtype_bytes: int = 2) -> int:
    """Token-payload bytes for a DeepEP kernel: first tensor of its enclosing
    FusedDispatch/FusedCombine range × dtype_bytes. 0 if no such range is found."""
    frame = _deepep_frame(stack)
    return _first_tensor_elems(frame.name) * dtype_bytes if frame else 0



def parse_range(text: str):
    ms = _SEND_RE.match(text)
    if ms:
        return "Send", int(ms.group(2))
    mr = _RECV_RE.match(text)
    if mr:
        return "Recv", int(mr.group(2))
    mb = _BASE_RE.match(text)
    if mb:
        return _OP_BY_BASE.get(mb.group(1), mb.group(1)), None
    return "?", None


class CommEvent(NamedTuple):
    op: str
    comm: str | None
    seq: int | None
    algo: str
    bytes: int
    peer: int | None
    gpu_dur: int
    gpu_start: int
    # the innermost non-plumbing enclosing NVTX scope (model phase)
    scope: str
    # Rank-local identity of the enclosing logical DeepEP call. None for NCCL.
    call_id: tuple[int, int] | None = None


def _innermost_range(ranges_by_tid, tid, ts):
    """
    Find the innermost nccl range on tid containing time ts (largest start).
    """
    arr = ranges_by_tid.get(tid)
    if not arr:
        return None
    i = bisect.bisect_right(arr, (ts, float("inf"))) - 1
    # scan back: nested ranges -> first with end>=ts is the innermost
    while i >= 0:
        s, e, idx = arr[i]
        if s <= ts <= e:
            return idx
        i -= 1
    return None


def load_comm_in_window(
    conn: sqlite3.Connection,
    idx: NvtxIndex,
    ws: int,
    we: int,
    dtype_bytes: int = 2,
) -> list[CommEvent]:
    """All comm CommEvents whose kernel starts in [ws, we] — NCCL + DeepEP.
    dtype_bytes sizes the marker-less volumes (P2P/DeepEP, no dtype); default bf16."""
    events: list[CommEvent] = []
    events.extend(_load_nccl(conn, idx, ws, we, dtype_bytes))
    events.extend(_load_deepeps(conn, idx, ws, we, dtype_bytes))
    return events


def _load_ranges(
    conn: sqlite3.Connection,
) -> tuple[dict[int, list], list[tuple[str, int | None, int]]]:
    """
    Load nccl ranges.
    """
    ranges = conn.execute(NCCL_RANGE_SQL).fetchall()
    ranges_by_tid: dict[int, list] = {}
    rmeta: list[tuple[str, int | None, int]] = []
    for idx, r in enumerate(ranges):
        op, peer = parse_range(r["text"])
        rmeta.append((op, peer, _range_elems(r["text"])))
        ranges_by_tid.setdefault(r["tid"], []).append((r["start"], r["end"], idx))
    for t in ranges_by_tid:
        ranges_by_tid[t].sort()
    return ranges_by_tid, rmeta


def _load_marks(
    conn: sqlite3.Connection, ranges_by_tid: dict[int, list]
) -> dict[int, list[tuple[int, dict]]]:
    """
    Load comm= detail markers.
    """
    marks = conn.execute(COMM_MARK_SQL).fetchall()
    range_marks: dict[int, list[tuple[int, dict]]] = {}
    for mk in marks:
        m = COMM_MARK_RE.match(mk["text"])
        if not m:
            continue
        ridx = _innermost_range(ranges_by_tid, mk["tid"], mk["start"])
        if ridx is not None:
            range_marks.setdefault(ridx, []).append((mk["start"], m.groupdict()))
    for ridx in range_marks:
        range_marks[ridx].sort()
    return range_marks


def _load_nccl(
    conn: sqlite3.Connection,
    idx: NvtxIndex,
    ws: int,
    we: int,
    dtype_bytes: int = 2,
) -> list[CommEvent]:
    """In-window ncclDevKernels; dtype_bytes sizes marker-less P2P send/recv.

    Comm semantics (op/comm/seq/algo/bytes/peer) come from the nccl:<op> ranges +
    comm= markers; the enclosing model scope comes from the NVTX stack via idx."""
    ranges_by_tid, rmeta = _load_ranges(conn)
    range_marks = _load_marks(conn, ranges_by_tid)
    kernels = conn.execute(NCCL_KERNEL_SQL, (ws, we)).fetchall()

    # resolve each kernel's model-level scope from its enclosing NVTX stack
    scope_by_id: dict[int, str] = {}
    stacks = idx.iter_stacks((k["api_start"], k["api_end"]) for k in kernels)
    for k, (_, _, stack) in zip(kernels, stacks):
        scope_by_id[id(k)] = _scope_from_stack(stack)

    # kernels bucketed to their innermost enclosing range
    range_kernels: dict[int, list] = {}
    for k in kernels:
        ridx = _innermost_range(ranges_by_tid, k["tid"], k["api_start"])
        if ridx is None:
            continue
        range_kernels.setdefault(ridx, []).append(k)
    for ridx in range_kernels:
        range_kernels[ridx].sort(key=lambda k: k["api_start"])

    events: list[CommEvent] = []
    for ridx, ks in range_kernels.items():
        op, peer, elems = rmeta[ridx]
        mks = range_marks.get(ridx, [])
        for i, k in enumerate(ks):
            md = mks[i][1] if i < len(mks) else None
            if md:
                comm = md["comm"]
                seq = int(md["seq"])
                algo = f"{md['algo']}/{md['proto']}"
                nbytes = int(md["count"]) * DTYPE_BYTES.get(md["dtype"], 1)
            else:
                comm = None
                seq = None
                algo = ""
                nbytes = elems * dtype_bytes
            events.append(
                CommEvent(
                    op,
                    comm,
                    seq,
                    algo,
                    nbytes,
                    peer,
                    k["gpu_dur"],
                    k["gpu_start"],
                    scope_by_id[id(k)],
                )
            )
    return events


def load_deepep_in_window(
    conn: sqlite3.Connection,
    idx: NvtxIndex,
    ws: int,
    we: int,
    dtype_bytes: int = 2,
) -> list[CommEvent]:
    """DeepEP-only CommEvents (Dispatch/Combine) whose kernel starts in [ws, we].

    Same as load_comm_in_window but skips the NCCL path entirely — for tools that
    only care about the MoE EP all-to-all (e.g. gpu-deepep-skew)."""
    return _load_deepeps(conn, idx, ws, we, dtype_bytes)


def _load_deepeps(
    conn: sqlite3.Connection,
    idx: NvtxIndex,
    ws: int,
    we: int,
    dtype_bytes: int = 2,
) -> list[CommEvent]:
    """One CommEvent per in-window DeepEP kernel (op Dispatch/Combine, comm=None);
    scope + token volume from the enclosing FusedDispatch/FusedCombine range."""
    kernels = conn.execute(DEEPEP_KERNEL_SQL, (ws, we)).fetchall()
    if not kernels:
        return []
    out: list[CommEvent] = []
    stacks = idx.iter_stacks((k["api_start"], k["api_end"]) for k in kernels)
    for k, (_, _, stack) in zip(kernels, stacks):
        name = k["kernel_name"]
        op = "Combine" if "combine" in name else "Dispatch"
        mode = "internode" if "internode" in k["full_name"] else "intranode"
        # only the data-moving dispatch/combine kernels carry payload; the
        # notify/layout helpers move no tokens (but their wait time is real).
        nbytes = (
            _deepep_volume(stack, dtype_bytes) if name in ("dispatch", "combine") else 0
        )
        frame = _deepep_frame(stack)
        out.append(
            CommEvent(
                op=op,
                comm=None,
                seq=None,
                algo=mode,
                bytes=nbytes,
                peer=None,
                gpu_dur=k["gpu_dur"],
                gpu_start=k["gpu_start"],
                scope=_scope_from_stack(stack),
                call_id=(frame.start, frame.end) if frame else None,
            )
        )
    return out


def group_nranks(conn: sqlite3.Connection) -> int:
    """Max NcclGroup:rank position on this rank (+1). Across all ranks this is the
    group size; on one rank it's only this rank's own max position."""
    mx = -1
    for (t,) in conn.execute(NCCL_GROUP_SQL):
        try:
            mx = max(mx, int(t.rsplit("=", 1)[1]))
        except ValueError:
            pass
    return mx + 1
