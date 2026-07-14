"""
This tool detects training-step boundaries (same detector as gpu-flame) and
exports a chosen step (or the whole post-warmup window) as a Chrome Trace Event
Format JSON — the format that chrome://tracing and the Perfetto UI ingest.

By default GPU and CPU are kept separate: GPU streams carry the kernels /
memcpy / memset, and NVTX ranges live on the CPU threads that issued them (their
true scope). Pass **--project** to instead resolve each GPU op's enclosing NVTX
stack (through its CUDA launch API interval — the exact kernel->NVTX mapping
gpu-flame uses) and nest it onto the op's GPU stream, so on the GPU timeline
every kernel sits inside the NVTX scopes that issued it (kernel = leaf).

Output: a top-level object with a ``traceEvents`` array of duration ("X") events
plus ``process_name``/``thread_name`` metadata ("M") events. Timestamps/durations
are microseconds and the origin is the trace start (MIN kernel start), so a step's
offset matches the gpu-flame step table.

Multiple ranks are merged into a single trace: each rank is given its own process
namespace (pid offset) and an "R<rank>" track-name prefix. Ranks are wall-clock
aligned by adding each file's session-start UTC epoch to its timestamps (event ts
is ns since that instant), so ranks captured on different nodes land on one real
timeline and cross-rank skew is visible; --no-align overlays them at t=0 instead.

Tracks:
  * GPU stream (default)     -> kernel/memcpy/memset, no NVTX
  * CPU thread (default)     -> NVTX push/pop ranges as issued on the host
  * GPU stream (--project)   -> NVTX scopes (projected) with kernel/memcpy/memset leaves
  * CPU thread (--cpu-nvtx)  -> add CPU-side NVTX on top of --project
  * CPU thread (default on)  -> CUDA runtime API calls (--no-cuda-api to drop)
  * launch flow (default on) -> click a GPU kernel -> its CPU launch site
                                (--no-flows to omit; --cuda-api for the exact API)

Author: yezhengmaolove@gmail.com
"""

import argparse
import gzip
import json
import re
import sqlite3
import sys
from collections import defaultdict

from ..utils.common import get_rank, open_db, require_kernel_table
from ..utils.nvtx import NvtxIndex
from .flamegraph import (
    TRACE_START_SQL,
    compute_step_windows,
    detect_steps,
    normalize,
    print_header_and_steps,
)

GPU_PID_BASE = 10_000_000  # keep GPU "pids" clear of real OS pids
MEM_TID_BASE = 1_000_000  # memcpy/memset sub-track offset in --flat mode
PID_RANK_STRIDE = 100_000_000  # per-rank pid namespace when merging many ranks
#   (> any real pid and > GPU_PID_BASE, so pid % stride recovers the GPU test)


def _flow_id(pid_offset: int, corr: int) -> int:
    """Per-rank-namespaced launch-flow id. correlationId is only unique within one
    profile, so merging ranks needs a rank prefix — but Perfetto's legacy-JSON flow
    binding wants a NUMERIC id (a string id silently fails to render), so encode it
    as rank_index * 1e10 + correlationId (corr well under 1e10)."""
    return (pid_offset // PID_RANK_STRIDE) * 10_000_000_000 + corr


# --- SQL --------------------------------------------------------------------
# GPU-op queries LEFT JOIN the launch-API interval so graph-launched ops (no
# runtime row) are not dropped. Each ends in "WHERE "; the caller appends the
# window predicate (kwin: k-qualified; win: bare start/end).
GPU_KERNEL_SQL = (
    "SELECT k.start,k.end,k.deviceId,k.streamId,r.start,r.end,"
    "k.demangledName,k.shortName,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,"
    "k.registersPerThread,k.staticSharedMemory,k.dynamicSharedMemory,k.correlationId,r.nameId "
    "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
    "LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=k.correlationId "
    "WHERE "
)
GPU_MEMCPY_SQL = (
    "SELECT k.start,k.end,k.deviceId,k.streamId,r.start,r.end,k.bytes,k.copyKind,k.correlationId "
    "FROM CUPTI_ACTIVITY_KIND_MEMCPY k "
    "LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=k.correlationId "
    "WHERE "
)
GPU_MEMSET_SQL = (
    "SELECT k.start,k.end,k.deviceId,k.streamId,r.start,r.end,k.bytes,k.correlationId "
    "FROM CUPTI_ACTIVITY_KIND_MEMSET k "
    "LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON r.correlationId=k.correlationId "
    "WHERE "
)
CPU_NVTX_SQL = (
    "SELECT start,end,text,textId,globalTid FROM NVTX_EVENTS "
    "WHERE end IS NOT NULL AND eventType IN (59,60) AND "
)
CUDA_API_SQL = (
    "SELECT start,end,globalTid,nameId,correlationId "
    "FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE end IS NOT NULL AND "
)
FLOW_API_SQL = (
    "SELECT start,globalTid,correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE "
)
FLOW_KERNEL_SQL = "SELECT start,deviceId,streamId,correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE "
# Cross-rank comm-flow markers: NCCL P2P (P2p:commId=…:seq=…:func=Send/Recv) and
# collectives (Op:comm=…:seq=…). commId/comm + seq is globally consistent across
# the ranks that participate, so it links the SAME communication across ranks.
COMM_FLOW_MARK_SQL = (
    "SELECT n.start, n.globalTid, COALESCE(n.text, s.value) t "
    "FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId "
    "WHERE (COALESCE(n.text,s.value) LIKE 'P2p:commId=0x%' "
    "OR COALESCE(n.text,s.value) LIKE '%:comm=0x%') AND n.start >= 0 AND "
)
# Metadata lookups (self-contained, no window).
STRINGS_SQL = "SELECT id, value FROM StringIds"
GPU_NAME_SQL = "SELECT id, name FROM TARGET_INFO_GPU"
MEMCPY_LABEL_SQL = "SELECT id, label FROM ENUM_CUDA_MEMCPY_OPER"
PROCESSES_SQL = "SELECT globalPid, name FROM PROCESSES"
THREAD_NAMES_SQL = "SELECT nameId, globalTid FROM ThreadNames"
# Session-start wall clock (UTC epoch ns). Event timestamps are ns since this
# instant, so utcEpochNs + ts is the event's absolute wall time — the basis for
# aligning ranks captured on different nodes onto one timeline.
SESSION_START_SQL = "SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME"


def decode(gid: int) -> tuple[int, int]:
    """nsys global id -> (pid, tid)."""
    return (gid >> 24) & 0xFFFFFF, gid & 0xFFFFFF


def _load_map(con, sql):
    return {r[0]: r[1] for r in con.execute(sql)}


def read_epoch(con):
    """Session-start UTC epoch (ns) for cross-file wall-clock alignment; None if
    the table is absent."""
    try:
        row = con.execute(SESSION_START_SQL).fetchone()
    except sqlite3.OperationalError:
        return None
    return row[0] if row else None


class TraceWriter:
    """Streams Chrome Trace Event Format JSON to a (optionally gzipped) file."""

    def __init__(self, path, t0, min_dur_ns):
        self.op = (
            gzip.open(path, "wt") if str(path).endswith(".gz") else open(path, "w")
        )
        self.op.write('{"displayTimeUnit":"ns","traceEvents":[\n')
        self.t0 = t0
        self.min_dur_ns = min_dur_ns
        self.n = 0
        self.procs: dict[int, str] = {}
        self.threads: dict[tuple[int, int], str] = {}
        # set per rank when merging several files into one trace
        self.pid_offset = 0
        self.name_prefix = ""
        # per-rank additive base (this file's UTC epoch) so ts_ns + time_offset
        # is absolute wall time; t0 is the shared absolute origin. Integer math
        # (both are large ns ints) keeps full precision before the /1000 float.
        self.time_offset = 0
        # per-rank-namespaced correlationId -> emitted (clamped) kernel start ns.
        # Same-stream starts are clamped forward during emission to keep ops serial
        # for strict Chrome nesting; launch flows must bind to that emitted start,
        # not the raw kernel start, or the flow lands outside the slice and Perfetto
        # can't connect it. Populated by x() as kernels are written, read by emit_flows.
        self.kslice: dict[str, int] = {}
        # cross-rank comm-flow endpoints: key (comm handle + seq, globally consistent
        # across the ranks in a collective/P2P) -> [(pid, tid, ts_us)]. Accumulated
        # across ALL ranks (not cleared), emitted once at the end.
        self.comm_pts: dict[tuple, list] = defaultdict(list)

    def x(self, pid, tid, name, ts_ns, dur_ns, cat, args_obj=None):
        if dur_ns < self.min_dur_ns:
            return
        e = {
            "ph": "X",
            "pid": pid + self.pid_offset,
            "tid": tid,
            "name": name,
            "cat": cat,
            "ts": (ts_ns + self.time_offset - self.t0) / 1000.0,
            "dur": dur_ns / 1000.0,
        }
        if args_obj:
            e["args"] = args_obj
        # Record this kernel's emitted (clamped) start so its launch flow binds to
        # the slice's real position rather than the raw correlationId time.
        if cat == "kernel" and args_obj and args_obj.get("correlationId") is not None:
            self.kslice[_flow_id(self.pid_offset, args_obj["correlationId"])] = ts_ns
        self._write(e)

    def raw(self, e):
        self._write(e)

    def _write(self, e):
        if self.n:
            self.op.write(",\n")
        self.op.write(json.dumps(e, ensure_ascii=False))
        self.n += 1

    def track(self, pid, pname, tid, tname):
        pid += self.pid_offset
        self.procs[pid] = self.name_prefix + pname
        self.threads[(pid, tid)] = tname

    def close(self):
        for pid, name in self.procs.items():
            self._write(
                {"ph": "M", "name": "process_name", "pid": pid, "args": {"name": name}}
            )
            self._write(
                {
                    "ph": "M",
                    "name": "process_sort_index",
                    "pid": pid,
                    # pid is offset per rank; recover the GPU test with % stride
                    "args": {
                        "sort_index": 1 if pid % PID_RANK_STRIDE >= GPU_PID_BASE else 0
                    },
                }
            )
        for (pid, tid), name in self.threads.items():
            self._write(
                {
                    "ph": "M",
                    "name": "thread_name",
                    "pid": pid,
                    "tid": tid,
                    "args": {"name": name},
                }
            )
        self.op.write("\n]}\n")
        self.op.close()


def _fetch_gpu_ops(con, ws, we, strings, memcpy_label):
    """All GPU ops in window as dicts: gpu_start/end, dev, stream, sort interval,
    name, cat, args. LEFT JOIN to runtime so graph-launched ops are not dropped."""
    # k-qualified: k JOIN r would make a bare 'start' ambiguous
    kwin = f"k.start < {we} AND k.end > {ws} AND k.start >= 0"
    ops = []
    for r in con.execute(GPU_KERNEL_SQL + kwin):
        name = strings.get(r[6]) or strings.get(r[7]) or "kernel"
        ops.append(
            {
                "gs": r[0],
                "ge": r[1],
                "dev": r[2] or 0,
                "stream": r[3],
                "ss": r[4] if r[4] is not None else r[0],
                "se": r[5] if r[5] is not None else r[1],
                "name": name,
                "cat": "kernel",
                "args": {
                    "grid": f"{r[8]}x{r[9]}x{r[10]}",
                    "block": f"{r[11]}x{r[12]}x{r[13]}",
                    "regs/thread": r[14],
                    "smem_static": r[15],
                    "smem_dyn": r[16],
                    "correlationId": r[17],
                    "launch_api": strings.get(r[18], ""),
                },
            }
        )
    for r in con.execute(GPU_MEMCPY_SQL + kwin):
        ops.append(
            {
                "gs": r[0],
                "ge": r[1],
                "dev": r[2] or 0,
                "stream": r[3],
                "ss": r[4] if r[4] is not None else r[0],
                "se": r[5] if r[5] is not None else r[1],
                "name": f"Memcpy {memcpy_label.get(r[7], r[7])}",
                "cat": "memcpy",
                "args": {"bytes": r[6], "correlationId": r[8]},
            }
        )
    for r in con.execute(GPU_MEMSET_SQL + kwin):
        ops.append(
            {
                "gs": r[0],
                "ge": r[1],
                "dev": r[2] or 0,
                "stream": r[3],
                "ss": r[4] if r[4] is not None else r[0],
                "se": r[5] if r[5] is not None else r[1],
                "name": "Memset",
                "cat": "memset",
                "args": {"bytes": r[6], "correlationId": r[7]},
            }
        )
    return ops


def emit_gpu_projected(
    con, idx, w, ws, we, strings, gpu_name, memcpy_label, stack_depth
):
    """Project each GPU op's enclosing NVTX stack onto its stream track (nested),
    with the op as the leaf — restoring the kernel<->NVTX relationship."""
    ops = _fetch_gpu_ops(con, ws, we, strings, memcpy_label)
    if not ops:
        return 0
    # resolve NVTX stack per op via its launch-API interval (needs ascending start)
    ops.sort(key=lambda o: o["ss"])
    for o, (_, _, stack) in zip(ops, idx.iter_stacks((o["ss"], o["se"]) for o in ops)):
        frames = [normalize(e.name) for e in stack]
        o["frames"] = frames[:stack_depth] if stack_depth > 0 else frames

    groups = defaultdict(list)
    for o in ops:
        groups[(o["dev"], o["stream"])].append(o)

    n0 = w.n
    for (dev, stream), os_ in sorted(groups.items()):
        pid = GPU_PID_BASE + dev
        w.track(
            pid, f"GPU {dev} ({gpu_name.get(dev, 'GPU')})", stream, f"Stream {stream}"
        )
        os_.sort(key=lambda o: o["gs"])
        # Start-driven construction: clamp each op's start forward so ops are serial
        # on the track (same-stream ops can overlap by a few us; that would break
        # strict Chrome nesting). Diverged NVTX frames close at the next op's start,
        # which mathematically guarantees each frame contains its children.
        open_frames: list[list] = []  # [name, start_ts]
        cursor = None
        for o in os_:
            s = o["gs"] if cursor is None else max(o["gs"], cursor)
            e = o["ge"] if o["ge"] > s else s
            P = o["frames"]
            cp = 0
            while cp < len(open_frames) and cp < len(P) and open_frames[cp][0] == P[cp]:
                cp += 1
            while len(open_frames) > cp:  # close diverged frames
                name, st = open_frames.pop()
                w.x(pid, stream, name, st, s - st, "nvtx")
            for d in range(cp, len(P)):  # open new frames
                open_frames.append([P[d], s])
            w.x(pid, stream, o["name"], s, e - s, o["cat"], o["args"])
            cursor = e
        while open_frames:  # close trailing frames
            name, st = open_frames.pop()
            w.x(pid, stream, name, st, cursor - st, "nvtx")
    return w.n - n0


def emit_gpu_flat(con, w, ws, we, gpu_name, memcpy_label, strings):
    """Flat GPU ops, no NVTX projection (mem on a sub-track). Ops are serialized
    per track (start clamped forward) so the few <=5us same-stream overlaps don't
    break strict nesting; kernels on stream tid, memcpy/memset on a mem sub-track."""
    groups = defaultdict(list)
    for o in _fetch_gpu_ops(con, ws, we, strings, memcpy_label):
        pid = GPU_PID_BASE + o["dev"]
        if o["cat"] == "kernel":
            tid, tname = o["stream"], f"Stream {o['stream']}"
        else:
            tid, tname = MEM_TID_BASE + o["stream"], f"Stream {o['stream']} (mem)"
        w.track(pid, f"GPU {o['dev']} ({gpu_name.get(o['dev'], 'GPU')})", tid, tname)
        groups[(pid, tid)].append(o)
    for (pid, tid), os_ in groups.items():
        os_.sort(key=lambda o: o["gs"])
        cursor = None
        for o in os_:
            s = o["gs"] if cursor is None else max(o["gs"], cursor)
            e = o["ge"] if o["ge"] > s else s
            w.x(pid, tid, o["name"], s, e - s, o["cat"], o["args"])
            cursor = e


def emit_cpu_nvtx(con, w, win, strings, proc_name, thread_name):
    for r in con.execute(CPU_NVTX_SQL + win):
        name = r[2] if r[2] is not None else strings.get(r[3], "")
        pid, tid = decode(r[4])
        w.track(
            pid,
            proc_name.get(pid, f"PID {pid}"),
            tid,
            thread_name.get(r[4]) or f"tid {tid}",
        )
        w.x(pid, tid, name or "nvtx", r[0], r[1] - r[0], "nvtx")


def emit_cuda_api(con, w, win, strings, proc_name, thread_name):
    for r in con.execute(CUDA_API_SQL + win):
        pid, tid = decode(r[2])
        w.track(
            pid,
            proc_name.get(pid, f"PID {pid}"),
            tid,
            thread_name.get(r[2]) or f"tid {tid}",
        )
        w.x(
            pid,
            tid,
            strings.get(r[3], "cudaApi"),
            r[0],
            r[1] - r[0],
            "cuda_api",
            {"correlationId": r[4]},
        )


_P2P_MARK_RE = re.compile(r"^P2p:commId=(0x[0-9a-f]+):rank=\d+:peer=\d+:seq=(\d+)")
_COLL_MARK_RE = re.compile(r"^(\w+):comm=(0x[0-9a-f]+):seq=(\d+)")


def collect_comm_flows(con, w, ws, we):
    """Record this rank's comm markers' (pid, tid, ts) keyed by the globally
    consistent (comm handle, seq) so emit_comm_flows can link them across ranks.
    Uses the marker directly — no kernel association."""
    seen: set = set()
    for start, gtid, text in con.execute(
        COMM_FLOW_MARK_SQL + f"n.start < {we} AND n.start >= {ws}"
    ):
        m = _P2P_MARK_RE.match(text)
        if m:
            key = ("p2p", m.group(1), int(m.group(2)))
        else:
            m = _COLL_MARK_RE.match(text)
            if not m:
                continue
            key = ("coll", m.group(1), m.group(2), int(m.group(3)))
        pid, tid = decode(gtid)
        pid += w.pid_offset
        if (key, pid) in seen:  # one endpoint per rank per key (drop push/pop dup)
            continue
        seen.add((key, pid))
        ts = (start + w.time_offset - w.t0) / 1000.0
        w.comm_pts[key].append((pid, tid, ts))


def emit_comm_flows(w):
    """Emit one cross-rank flow per comm key that has >= 2 endpoints in the trace
    (a communication whose peers are all present); skip the rest. Distinct id
    range and cat='comm' so it never collides with the launch flows."""
    fid = 500_000_000_000_000  # above launch-flow ids, below 2^53
    n = 0
    for pts in w.comm_pts.values():
        if len(pts) < 2:  # peer not in this trace -> nothing to connect
            continue
        pts.sort(key=lambda p: p[2])  # by time; s -> t… -> f draws the chain
        last = len(pts) - 1
        for i, (pid, tid, ts) in enumerate(pts):
            w.raw(
                {
                    "ph": "s" if i == 0 else ("f" if i == last else "t"),
                    "pid": pid,
                    "tid": tid,
                    "name": "comm",
                    "cat": "comm",
                    "id": fid,
                    "bp": "e",
                    "ts": ts,
                }
            )
        fid += 1
        n += 1
    return n


def emit_flows(con, w, win):
    api = {}
    for r in con.execute(FLOW_API_SQL + win):
        pid, tid = decode(r[1])
        api[r[2]] = (r[0], pid, tid)
    for r in con.execute(FLOW_KERNEL_SQL + win):
        a = api.get(r[3])
        if not a:
            continue
        gpid = GPU_PID_BASE + (r[1] or 0)
        # Flow id MUST be namespaced per rank (correlationId only unique within one
        # profile, else merged ranks cross-link) but stay NUMERIC (Perfetto's
        # legacy-JSON flow binding drops string ids).
        flow_id = _flow_id(w.pid_offset, r[3])
        # GPU end must sit on the kernel's EMITTED (clamped) start, not the raw
        # k.start — else for same-stream-overlapping kernels the flow lands just
        # before the slice and won't bind. Skip if the kernel wasn't emitted.
        f_ts = w.kslice.get(flow_id)
        if f_ts is None:
            continue
        for ph, pid, tid, ts in (("s", a[1], a[2], a[0]), ("f", gpid, r[2], f_ts)):
            # bp="e" binds each endpoint to its *enclosing* slice: the CPU launch
            # site (cudaLaunchKernel with --cuda-api, else the enclosing NVTX op)
            # and the GPU kernel — so clicking the kernel jumps to its launcher.
            w.raw(
                {
                    "ph": ph,
                    "pid": pid + w.pid_offset,
                    "tid": tid,
                    "name": "launch",
                    "cat": "launch",
                    "id": flow_id,
                    "bp": "e",
                    "ts": (ts + w.time_offset - w.t0) / 1000.0,
                }
            )


def export_trace(con, idx, w, ws, we, args):
    """Emit one rank's window into the (already-open) writer w. The caller sets
    w.pid_offset / w.name_prefix per rank so many ranks share one trace. Returns
    (events, tracks) added by this rank."""
    strings = _load_map(con, STRINGS_SQL)
    gpu_name = _load_map(con, GPU_NAME_SQL)
    memcpy_label = _load_map(con, MEMCPY_LABEL_SQL)
    proc_name = {}
    for gpid, name in con.execute(PROCESSES_SQL):
        proc_name[(gpid >> 24) & 0xFFFFFF] = name
    thread_name = {}
    for name_id, gtid in con.execute(THREAD_NAMES_SQL):
        thread_name[gtid] = strings.get(name_id, "")

    # events that intersect [ws, we]; start>=0 drops nsys's stray pre-session range
    win = f"start < {we} AND end > {ws} AND start >= 0"

    w.kslice.clear()  # emitted-kernel-start map is per rank
    ev0, trk0 = w.n, len(w.procs)
    if args.project:
        # Opt-in: nest each kernel under its NVTX stack on the GPU stream.
        emit_gpu_projected(
            con, idx, w, ws, we, strings, gpu_name, memcpy_label, args.stack_depth
        )
        if args.cpu_nvtx:
            emit_cpu_nvtx(con, w, win, strings, proc_name, thread_name)
    else:
        # Default: GPU streams carry kernels only; NVTX lives on CPU threads,
        # its true (issuing) scope. --no-cpu-nvtx drops the CPU side (GPU-only).
        emit_gpu_flat(con, w, ws, we, gpu_name, memcpy_label, strings)
        if not args.no_cpu_nvtx:
            emit_cpu_nvtx(con, w, win, strings, proc_name, thread_name)
    if args.cuda_api:
        emit_cuda_api(con, w, win, strings, proc_name, thread_name)
    if args.flows:
        emit_flows(con, w, win)
    if args.comm_flows:
        collect_comm_flows(con, w, ws, we)
    return w.n - ev0, len(w.procs) - trk0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect training steps and export a step (or the post-warmup "
        "window) as Chrome/Perfetto trace JSON. Default splits GPU and CPU: "
        "GPU streams carry kernels, NVTX stays on CPU threads. Use --project "
        "to nest each kernel under its NVTX scope on the GPU timeline instead."
    )
    p.add_argument(
        "db",
        nargs="+",
        metavar="db.sqlite",
        help="One or more .sqlite profiles. Multiple ranks are merged into a "
        "single trace (each rank in its own process namespace, prefixed "
        "'R<rank>', on a shared time origin so same-node ranks line up).",
    )
    p.add_argument(
        "--step-nvtx",
        default="Optimizer.step",
        metavar="SUBSTR",
        help="NVTX substring marking one training-step boundary "
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
        "--step",
        type=int,
        default=None,
        metavar="N",
        help="Export only step N (1-based, as printed). Omit for the "
        "whole post-warmup window.",
    )
    p.add_argument(
        "-o",
        "--output",
        default=None,
        metavar="OUT",
        help="Write the trace here (.json or .json.gz; .gz recommended). "
        "Omit to just print the step table.",
    )
    p.add_argument(
        "--stack-depth",
        type=int,
        default=0,
        metavar="N",
        help="Limit projected NVTX nesting depth (outermost-first); 0 = full (default)",
    )
    p.add_argument(
        "--project",
        action="store_true",
        help="Project each kernel's NVTX stack onto its GPU stream (nested, "
        "kernel = leaf). Default is the split view: GPU streams carry "
        "kernels only, NVTX stays on CPU threads (its issuing scope).",
    )
    p.add_argument(
        "--cpu-nvtx",
        action="store_true",
        help="With --project, also emit NVTX on CPU threads (in the default "
        "split view CPU NVTX is always emitted)",
    )
    p.add_argument(
        "--no-cpu-nvtx",
        action="store_true",
        help="In the default split view, drop the CPU-side NVTX tracks — GPU "
        "streams (kernels only), no CPU side (much smaller for many ranks)",
    )
    p.add_argument(
        "--cuda-api",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit CUDA runtime API calls (cudaLaunchKernel, …) on CPU threads "
        "(default on) so launch flows land on the exact launch slice. Adds a lot "
        "of events; --no-cuda-api drops them (flows then land on the enclosing "
        "NVTX op).",
    )
    p.add_argument(
        "--flows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit CPU-launch -> GPU-kernel flow arrows (default on): click a GPU "
        "kernel in Perfetto to jump to its launch site — the exact cudaLaunchKernel "
        "(or the enclosing NVTX op under --no-cuda-api). Use --no-flows to omit.",
    )
    p.add_argument(
        "--comm-flows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emit CROSS-RANK communication flows (default on): NCCL P2P and "
        "collectives are linked across ranks by their (comm handle, seq) marker — "
        "click a send to jump to the matching recv on the peer rank. Only drawn "
        "when both endpoints are in the trace. --no-comm-flows to omit.",
    )
    p.add_argument(
        "--no-align",
        action="store_true",
        help="When merging several ranks, do NOT wall-clock align them; overlay "
        "all at t=0. Default aligns by each file's session-start UTC epoch so "
        "ranks on different nodes line up (cross-node clocks track NTP).",
    )
    p.add_argument(
        "--min-dur-ns",
        type=int,
        default=0,
        metavar="NS",
        help="Drop events shorter than this (shrinks output)",
    )
    args = p.parse_args()
    if args.skip_steps < 0:
        p.error("--skip-steps must be >= 0")
    if args.stack_depth < 0:
        p.error("--stack-depth must be >= 0")
    return args


def prepare_rank(path, args):
    """Open one rank, detect steps, print its table, and resolve the export
    window. Returns a context dict, or None if the file has no usable window
    (message printed to stderr; other ranks continue)."""
    conn = open_db(path)
    require_kernel_table(conn, path)
    rank = get_rank(conn)

    idx = NvtxIndex(conn, rank)
    steps = detect_steps(idx, args.step_nvtx)
    if len(steps) <= args.skip_steps:
        conn.close()
        print(
            f"Error [{path}]: found {len(steps)} NVTX step markers matching "
            f"'{args.step_nvtx}', cannot skip {args.skip_steps}; skipping file.",
            file=sys.stderr,
        )
        return None

    t_min = conn.execute(TRACE_START_SQL).fetchone()[0]
    step_windows, window_start, window_end = compute_step_windows(
        steps, t_min, args.skip_steps
    )
    print_header_and_steps(
        path,
        rank,
        args.step_nvtx,
        steps,
        step_windows,
        args.skip_steps,
        t_min,
        window_end - window_start,
    )

    if args.step is not None:
        if not (1 <= args.step <= len(step_windows)):
            conn.close()
            print(
                f"Error [{path}]: --step {args.step} out of range "
                f"(1..{len(step_windows)}); skipping file.",
                file=sys.stderr,
            )
            return None
        ws, we, sname = step_windows[args.step - 1]
        sel = f"step {args.step} ({sname})"
        if args.step <= args.skip_steps:
            print(f"Warning: step {args.step} is within the skipped warmup.")
    else:
        ws, we = window_start, window_end
        sel = f"post-warmup steps {args.skip_steps + 1}..{len(steps)}"

    return {
        "path": path,
        "conn": conn,
        "idx": idx,
        "rank": rank,
        "ws": ws,
        "we": we,
        "t_min": t_min,
        "epoch": read_epoch(conn),
        "sel": sel,
    }


if __name__ == "__main__":
    args = parse_args()

    ranks = [c for c in (prepare_rank(p, args) for p in args.db) if c is not None]
    if not ranks:
        sys.exit(1)

    if not args.output:
        for c in ranks:
            span = (c["we"] - c["ws"]) / 1e6
            print(
                f"Selected [{c['path']}]: {c['sel']}  "
                f"[{(c['ws'] - c['t_min']) / 1e6:.2f} -> "
                f"{(c['we'] - c['t_min']) / 1e6:.2f} ms, {span:.2f} ms]"
            )
            c["conn"].close()
        print("(no --output given; step table only. Pass -o out.json.gz to export.)")
        sys.exit(0)

    # Merge every rank into one trace. To place ranks on a common timeline we add
    # each file's session-start UTC epoch to its timestamps (event ts is ns since
    # that instant), so ranks captured on different nodes — with unrelated raw
    # clocks — align to real wall time. Without alignment they'd just overlay at
    # t=0, hiding the (here up to seconds) capture-start stagger between ranks.
    multi = len(ranks) > 1
    align = not args.no_align and all(c["epoch"] is not None for c in ranks)
    if not align and not args.no_align and multi:
        print(
            "Warning: session-start epoch missing on some file(s); overlaying at "
            "t=0 without wall-clock alignment.",
            file=sys.stderr,
        )
    for c in ranks:
        c["base"] = c["epoch"] if align else 0
    t0 = min(c["base"] + c["t_min"] for c in ranks)
    stagger = max(c["base"] + c["t_min"] for c in ranks) - t0

    mode = "NVTX-projected" if args.project else "split (GPU / CPU-NVTX)"
    print(f"Exporting {len(ranks)} rank(s) ({mode}) -> {args.output}")
    if multi:
        if align:
            print(
                f"  time: UTC-epoch aligned (capture-start stagger across ranks "
                f"= {stagger / 1e9:.3f} s)"
            )
        else:
            print(
                "  time: NOT aligned — overlaid at t=0"
                + (" (--no-align)" if args.no_align else "")
            )

    w = TraceWriter(args.output, t0, args.min_dur_ns)
    total_ev = 0
    for i, c in enumerate(ranks):
        w.pid_offset = i * PID_RANK_STRIDE
        w.time_offset = c["base"]
        if not multi:
            w.name_prefix = ""
        elif c["rank"] is not None:
            w.name_prefix = f"R{c['rank']} "
        else:
            w.name_prefix = f"[{i}] "
        n_ev, n_trk = export_trace(c["conn"], c["idx"], w, c["ws"], c["we"], args)
        total_ev += n_ev
        print(f"  {c['path']}: {c['sel']} -> {n_ev} events, {n_trk} tracks")
        c["conn"].close()
    if args.comm_flows:
        n_comm = emit_comm_flows(w)
        total_ev += 2 * n_comm  # rough; s/f (+t) events
        print(f"  comm flows (cross-rank NCCL P2P/collective): {n_comm}")
    n_trk_all = len(w.procs)
    w.close()
    print(f"Wrote {total_ev} events across {n_trk_all} tracks -> {args.output}")
    print("Open at https://ui.perfetto.dev (drag & drop; .gz is fine).")
