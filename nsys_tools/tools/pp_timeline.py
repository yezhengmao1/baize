"""gpu-sched: op-level 1F1B pipeline timeline (interleaved/vpp) SVG.

Unlike ``gpu-groups --gbs`` (which draws a *unit-time* F/B Gantt purely to show the
schedule shape), this tool draws a **wall-clock-scaled** timeline where every
forward/backward op of every microbatch is decomposed into its measured GPU
sub-phases and each sub-phase box is as wide as its real duration.  The schedule
order is Megatron's interleaved 1F1B (reused from ``parallel_groups._pp_program``);
op start times come from the data dependencies plus each device's op durations.

Phase legend (one colored box per sub-op):

  forward   F  attn fwd      D  dispatch(EP)   E  moe-mlp fwd   C  combine(EP)
            M  dense-mlp fwd V  embedding      L  lm head
  backward  F^D/F^W attn d/w-grad   E^D/E^W expert d/w-grad   M^D/M^W dense d/w-grad
            D^ dispatch-bwd  C^ combine-bwd  V^ embed-bwd  L^ lm-head-bwd

A transformer layer ('t') is MoE (F,D,E,C forward) unless its global id is in
``--dense-layers`` (F,M forward) — by default only layer 0, per the given model.
Per-phase times come from built-in DEFAULTS (a measured A3B-8EP-4PP run), overridden
by a ``--config`` JSON (keys = phase tokens, e.g. {"F": 1.4, "E^D": 1.9}) and/or the
per-phase ``--t-<phase>`` CLI flags.  Precedence: ``--t-*`` flag > ``--config`` >
default — no CSV; re-time the whole picture by hand via params or a config file.
"""

import argparse
import json
import sys
from pathlib import Path

from .parallel_groups import PPLayout, _pp_program

# --- phase catalog ------------------------------------------------------------
# token -> (family, csv component name or None, empirical default ms, human label)
# family drives the color hue; forward tokens are the bare letters, backward
# tokens carry ^D / ^W / ^ suffixes.
FWD_TOKENS = ["V", "F", "M", "D", "E", "C", "L"]
BWD_TOKENS = ["L^", "C^", "E^W", "E^D", "D^", "M^W", "M^D", "F^W", "F^D", "V^"]

# the model op behind each phase token (informational — shown in the summary)
SRC = {
    "F": "attn_fwd",
    "M": "dmlp_fwd",
    "D": "dispatch_fwd",
    "E": "expert_fwd (+router)",
    "C": "combine_fwd",
    "V": "embedding_fwd",
    "L": "lm_head_fwd (+loss)",
    "F^D": "attn_dgrad",
    "F^W": "attn_wgrad",
    "E^D": "expert_dgrad",
    "E^W": "expert_wgrad",
    "M^D": "dmlp_dgrad",
    "M^W": "dmlp_wgrad",
    "D^": "dispatch_bwd",
    "C^": "combine_bwd",
    "V^": "embedding_bwd",
    "L^": "lm_head_bwd (+loss)",
}
# built-in default ms per phase token — a measured A3B-8EP-4PP-VPP1 run
# (router already folded into E; loss folded into L / L^). Override with
# --config JSON (same token keys) or --t-<phase> flags.
DEFAULTS = {
    "V": 0.0129,
    "F": 1.4037,
    "M": 0.6808,
    "D": 1.8637,
    "E": 1.5908,  # expert_fwd 1.3262 + router 0.2646
    "C": 1.5724,
    "L": 11.0927,  # lm_head_fwd 7.2246 + loss_fwd 3.8681
    "L^": 16.7053,  # lm_head_bwd 14.0659 + loss_bwd 2.6394
    "C^": 1.9165,
    "E^W": 2.6714,
    "E^D": 1.9103,
    "D^": 1.9155,
    "M^W": 1.9869,
    "M^D": 0.7742,
    "F^W": 0.4762,
    "F^D": 2.6872,
    "V^": 0.4151,
}

# family hue / label used for coloring + legend
FAMILY = {
    "F": ("attn", 212),
    "F^D": ("attn", 212),
    "F^W": ("attn", 212),
    "M": ("dense", 32),
    "M^D": ("dense", 32),
    "M^W": ("dense", 32),
    "D": ("disp", 174),
    "D^": ("disp", 174),
    "E": ("moe", 142),
    "E^D": ("moe", 142),
    "E^W": ("moe", 142),
    "C": ("comb", 280),
    "C^": ("comb", 280),
    "V": ("embed", 222),
    "V^": ("embed", 222),
    "L": ("lmhead", 350),
    "L^": ("lmhead", 350),
}
LABEL = {
    "F": "F  attn fwd",
    "D": "D  dispatch (EP)",
    "E": "E  moe-mlp fwd",
    "C": "C  combine (EP)",
    "M": "M  dense-mlp fwd",
    "V": "V  embedding",
    "L": "L  lm head",
    "F^D": "Fᴰ  attn dgrad",
    "F^W": "Fᵂ  attn wgrad",
    "E^D": "Eᴰ  expert dgrad",
    "E^W": "Eᵂ  expert wgrad",
    "M^D": "Mᴰ  dense dgrad",
    "M^W": "Mᵂ  dense wgrad",
    "D^": "Dʸ  dispatch bwd",
    "C^": "Cʸ  combine bwd",
    "V^": "Vʸ  embed bwd",
    "L^": "Lʸ  lm-head bwd",
}
# short in-box glyph (with unicode super-scripts for d/w-grad)
GLYPH = {
    "F^D": "Fᴰ",
    "F^W": "Fᵂ",
    "E^D": "Eᴰ",
    "E^W": "Eᵂ",
    "M^D": "Mᴰ",
    "M^W": "Mᵂ",
    "D^": "Dʸ",
    "C^": "Cʸ",
    "V^": "Vʸ",
    "L^": "Lʸ",
}


def load_config(path):
    """Read a per-phase time config (JSON: {token: ms}); {} if no path.

    Keys are phase tokens (F, D, E, C, M, V, L, F^D, E^W, D^, ...) — the same
    names shown in the legend and used by the --t-<phase> flags. Unknown keys are
    rejected so typos surface instead of silently doing nothing."""
    if not path:
        return {}
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        sys.exit(f"--config {path}: expected a JSON object {{token: ms}}")
    valid = set(FWD_TOKENS + BWD_TOKENS)
    cfg = {}
    for k, v in raw.items():
        if k not in valid:
            sys.exit(
                f"--config {path}: unknown phase token {k!r}; "
                f"valid tokens: {sorted(valid)}"
            )
        cfg[k] = float(v)
    return cfg


def resolve_times(args, config):
    """Final ms per phase token. Precedence: --t-<phase> flag > --config > DEFAULTS.

    Values are per-box totals as drawn (router already folded into E, loss into
    L / L^) — set them directly; there is no separate router/loss knob."""
    t = {}
    for tok in FWD_TOKENS + BWD_TOKENS:
        override = getattr(args, "t_" + _flag(tok), None)
        if override is not None:
            t[tok] = override
        elif tok in config:
            t[tok] = config[tok]
        else:
            t[tok] = DEFAULTS[tok]
    return t


def _flag(tok):
    """Token -> CLI-flag-safe suffix: F^D -> Fd, E^W -> Ew, D^ -> Db, V -> V."""
    if tok.endswith("^D"):
        return tok[0] + "d"
    if tok.endswith("^W"):
        return tok[0] + "w"
    if tok.endswith("^"):
        return tok[0] + "b"
    return tok


# --- per-chunk sub-phase decomposition ---------------------------------------
def chunk_segments(chunk, dense_set, t):
    """(fwd_segs, bwd_segs) for one layout chunk = lists of (token, ms).

    Forward walks the chunk's chars in order (E, layers, L); backward is the exact
    reverse computation (L^, layers reversed, V^), each layer emitting its grads.
    """
    fwd, layers_seq = [], []
    lids = list(chunk["layers"])
    li = 0
    for c in chunk["chars"]:
        if c == "E":
            fwd.append(("V", t["V"]))
        elif c == "L":
            fwd.append(("L", t["L"]))
        elif c in ("t", "m"):
            gid = lids[li] if li < len(lids) else None
            li += 1
            dense = gid in dense_set
            layers_seq.append(dense)
            if dense:
                fwd += [("F", t["F"]), ("M", t["M"])]
            else:
                fwd += [("F", t["F"]), ("D", t["D"]), ("E", t["E"]), ("C", t["C"])]
    bwd = []
    if any(c == "L" for c in chunk["chars"]):
        bwd.append(("L^", t["L^"]))
    for dense in reversed(layers_seq):
        if dense:
            bwd += [
                ("M^D", t["M^D"]),
                ("M^W", t["M^W"]),
                ("F^D", t["F^D"]),
                ("F^W", t["F^W"]),
            ]
        else:
            bwd += [
                ("C^", t["C^"]),
                ("E^D", t["E^D"]),
                ("E^W", t["E^W"]),
                ("D^", t["D^"]),
                ("F^D", t["F^D"]),
                ("F^W", t["F^W"]),
            ]
    if any(c == "E" for c in chunk["chars"]):
        bwd.append(("V^", t["V^"]))
    return fwd, bwd


def build_durations(layout, dense_set, t):
    """Per virtual-stage vs = chunk*pp + pp_rank: (fwd_segs, bwd_segs).

    layout.by_pp[d][c] is pp-rank d's vpp-chunk c (vpp-major id = c*pp + d)."""
    p, v = layout.pp, layout.vpp
    seg_f, seg_b = {}, {}
    for c in range(v):
        for d in range(p):
            fwd, bwd = chunk_segments(layout.by_pp[d][c], dense_set, t)
            seg_f[c * p + d] = fwd
            seg_b[c * p + d] = bwd
    return seg_f, seg_b


# --- dependency-timed 1F1B schedule (real durations) --------------------------
def schedule_timed(p, v, m, dur_f, dur_b):
    """Interleaved 1F1B op timeline with wall-clock durations.

    Same program order as parallel_groups._pp_program (Megatron interleaved 1F1B,
    microbatch_group_size = pp); each op starts at max(device-free, data-dep-ready)
    and lasts dur_f[vs]/dur_b[vs].  Returns (ops, makespan, busy_per_stage) with
    each op {kind, mb, chunk, stage, vs, t, dur, phase}."""
    V = p * v
    programs = [_pp_program(p, v, m, d) for d in range(p)]
    ptr = [0] * p
    dev_t = [0.0] * p
    busy = [0.0] * p
    end = {}  # (kind, mb, vs) -> finish time
    ops = []
    total = sum(len(pr) for pr in programs)
    guard = 0
    while len(ops) < total:
        progressed = False
        for d in range(p):
            if ptr[d] >= len(programs[d]):
                continue
            kind, c, mb, phase = programs[d][ptr[d]]
            vs = c * p + d
            if kind == "F":
                if vs > 0 and ("F", mb, vs - 1) not in end:
                    continue
                dep = end.get(("F", mb, vs - 1), 0.0) if vs > 0 else 0.0
                dur = dur_f[vs]
            else:
                if ("F", mb, vs) not in end:
                    continue
                if vs < V - 1 and ("B", mb, vs + 1) not in end:
                    continue
                dep = max(
                    end.get(("B", mb, vs + 1), 0.0) if vs < V - 1 else 0.0,
                    end[("F", mb, vs)],
                )
                dur = dur_b[vs]
            t0 = max(dev_t[d], dep)
            end[(kind, mb, vs)] = t0 + dur
            dev_t[d] = t0 + dur
            busy[d] += dur
            ops.append(
                {
                    "kind": kind,
                    "mb": mb,
                    "chunk": c,
                    "stage": d,
                    "vs": vs,
                    "t": t0,
                    "dur": dur,
                    "phase": phase,
                }
            )
            ptr[d] += 1
            progressed = True
        if not progressed:
            guard += 1
            if guard > total + 10:
                break
    makespan = max((o["t"] + o["dur"] for o in ops), default=0.0)
    return ops, makespan, busy


# --- EP-overlap schedule: two streams/device, Megatron combined-1F1B ----------
# Each MoE layer is 4 nodes on 2 streams (per model_chunk_schedule_plan.py):
#   comp: attn -> mlp(expert)          comm: dispatch(A2A) -> combine(A2A)
# Backward mirrors it. In steady 1F1B a forward mb and a backward mb are paired
# into one combined step whose node issue order (per combined_1f1b.py) interleaves
# the two so the forward A2A hides behind the backward compute and vice versa:
#   comm: combine_bwd | dispatch_fwd->dispatch_bwd | combine_fwd
#   comp: attn_fwd    | mlp_bwd->mlp_bwd_dw->mlp_fwd | attn_bwd
# Two in-order CUDA streams + cross-stream data deps are then list-scheduled, so
# comm that overruns the compute it hides under shows up as exposed comp-stream gaps.
class _NodeGen:
    def __init__(self):
        self.nodes = []
        self._i = 0

    def mk(self, tok, stream, ms, mb, vs, kind, dev):
        n = {
            "id": self._i,
            "dev": dev,
            "stream": stream,
            "tok": tok,
            "ms": ms,
            "mb": mb,
            "vs": vs,
            "kind": kind,
            "preds": [],
        }
        self._i += 1
        self.nodes.append(n)
        return n


def _chunk_layer_dense(chunk, dense_set):
    """List of per-'t'-layer dense flags, in written (forward) order."""
    out, lids, li = [], list(chunk["layers"]), 0
    for ch in chunk["chars"]:
        if ch in ("t", "m"):
            gid = lids[li] if li < len(lids) else None
            li += 1
            out.append(gid in dense_set)
    return out


def _fwd_op(g, mb, vs, layout, dense_set, t):
    d = vs % layout.pp
    chunk = layout.by_pp[d][vs // layout.pp]
    has_E = any(c == "E" for c in chunk["chars"])
    has_L = any(c == "L" for c in chunk["chars"])
    pre = g.mk("V", "comp", t["V"], mb, vs, "F", d) if has_E else None
    layers = []
    for dense in _chunk_layer_dense(chunk, dense_set):
        if dense:
            layers.append(
                {
                    "moe": False,
                    "attn": g.mk("F", "comp", t["F"], mb, vs, "F", d),
                    "mlp": g.mk("M", "comp", t["M"], mb, vs, "F", d),
                    "dispatch": None,
                    "combine": None,
                }
            )
        else:
            layers.append(
                {
                    "moe": True,
                    "attn": g.mk("F", "comp", t["F"], mb, vs, "F", d),
                    "dispatch": g.mk("D", "comm", t["D"], mb, vs, "F", d),
                    "mlp": g.mk("E", "comp", t["E"], mb, vs, "F", d),
                    "combine": g.mk("C", "comm", t["C"], mb, vs, "F", d),
                }
            )
    post = g.mk("L", "comp", t["L"], mb, vs, "F", d) if has_L else None
    # data deps along the activation path: (V) -> attn->disp->mlp->comb -> ... -> (L)
    prev = pre
    for L in layers:
        if prev:
            L["attn"]["preds"].append(prev["id"])
        if L["moe"]:
            L["dispatch"]["preds"].append(L["attn"]["id"])
            L["mlp"]["preds"].append(L["dispatch"]["id"])
            L["combine"]["preds"].append(L["mlp"]["id"])
            prev = L["combine"]
        else:
            L["mlp"]["preds"].append(L["attn"]["id"])
            prev = L["mlp"]
    if post:
        post["preds"].append(prev["id"])
        prev = post
    first = pre or layers[0]["attn"]
    return {"layers": layers, "pre": pre, "post": post, "first": first, "last": prev}


def _bwd_op(g, mb, vs, layout, dense_set, t):
    d = vs % layout.pp
    chunk = layout.by_pp[d][vs // layout.pp]
    has_E = any(c == "E" for c in chunk["chars"])
    has_L = any(c == "L" for c in chunk["chars"])
    pre = g.mk("L^", "comp", t["L^"], mb, vs, "B", d) if has_L else None
    layers = []
    for dense in reversed(_chunk_layer_dense(chunk, dense_set)):
        if dense:
            layers.append(
                {
                    "moe": False,
                    "combine_bwd": None,
                    "dispatch_bwd": None,
                    "mlp_dgrad": g.mk("M^D", "comp", t["M^D"], mb, vs, "B", d),
                    "mlp_wgrad": g.mk("M^W", "comp", t["M^W"], mb, vs, "B", d),
                    "attn_dgrad": g.mk("F^D", "comp", t["F^D"], mb, vs, "B", d),
                    "attn_wgrad": g.mk("F^W", "comp", t["F^W"], mb, vs, "B", d),
                }
            )
        else:
            layers.append(
                {
                    "moe": True,
                    "combine_bwd": g.mk("C^", "comm", t["C^"], mb, vs, "B", d),
                    "mlp_dgrad": g.mk("E^D", "comp", t["E^D"], mb, vs, "B", d),
                    "mlp_wgrad": g.mk("E^W", "comp", t["E^W"], mb, vs, "B", d),
                    "dispatch_bwd": g.mk("D^", "comm", t["D^"], mb, vs, "B", d),
                    "attn_dgrad": g.mk("F^D", "comp", t["F^D"], mb, vs, "B", d),
                    "attn_wgrad": g.mk("F^W", "comp", t["F^W"], mb, vs, "B", d),
                }
            )
    post = g.mk("V^", "comp", t["V^"], mb, vs, "B", d) if has_E else None
    # dgrad path: (L^) -> comb^->mlp_dgrad->disp^->attn_dgrad -> ... -> (V^);
    # wgrads hang off their dgrad (delay_wgrad-style, off the critical path)
    prev = pre
    for L in layers:
        if L["moe"]:
            if prev:
                L["combine_bwd"]["preds"].append(prev["id"])
            L["mlp_dgrad"]["preds"].append(L["combine_bwd"]["id"])
            L["mlp_wgrad"]["preds"].append(L["mlp_dgrad"]["id"])
            L["dispatch_bwd"]["preds"].append(L["mlp_dgrad"]["id"])
            L["attn_dgrad"]["preds"].append(L["dispatch_bwd"]["id"])
            L["attn_wgrad"]["preds"].append(L["attn_dgrad"]["id"])
            prev = L["attn_dgrad"]
        else:
            if prev:
                L["mlp_dgrad"]["preds"].append(prev["id"])
            L["mlp_wgrad"]["preds"].append(L["mlp_dgrad"]["id"])
            L["attn_dgrad"]["preds"].append(L["mlp_dgrad"]["id"])
            L["attn_wgrad"]["preds"].append(L["attn_dgrad"]["id"])
            prev = L["attn_dgrad"]
    if post:
        post["preds"].append(prev["id"])
        prev = post
    first = pre or layers[0]["combine_bwd"] or layers[0]["mlp_dgrad"]
    return {"layers": layers, "pre": pre, "post": post, "first": first, "last": prev}


def _issue_fwd(comp, comm, op):
    if op["pre"]:
        comp.append(op["pre"]["id"])
    for L in op["layers"]:
        comp.append(L["attn"]["id"])
        if L["dispatch"]:
            comm.append(L["dispatch"]["id"])
        comp.append(L["mlp"]["id"])
        if L["combine"]:
            comm.append(L["combine"]["id"])
    if op["post"]:
        comp.append(op["post"]["id"])


def _issue_bwd(comp, comm, op):
    if op["pre"]:
        comp.append(op["pre"]["id"])
    for L in op["layers"]:
        if L["combine_bwd"]:
            comm.append(L["combine_bwd"]["id"])
        comp.append(L["mlp_dgrad"]["id"])
        comp.append(L["mlp_wgrad"]["id"])
        if L["dispatch_bwd"]:
            comm.append(L["dispatch_bwd"]["id"])
        comp.append(L["attn_dgrad"]["id"])
        comp.append(L["attn_wgrad"]["id"])
    if op["post"]:
        comp.append(op["post"]["id"])


def _issue_combined(comp, comm, F, B):
    """Interleave a forward op F and backward op B per combined_1f1b.py so each
    A2A comm node is issued alongside a compute node from the other direction."""
    if F["pre"]:
        comp.append(F["pre"]["id"])  # embedding fwd
    if B["pre"]:
        comp.append(B["pre"]["id"])  # lm-head bwd
    nf, nb = len(F["layers"]), len(B["layers"])
    for i in range(max(nf, nb)):
        fl = F["layers"][i] if i < nf else None
        bl = B["layers"][i] if i < nb else None
        if fl:
            comp.append(fl["attn"]["id"])  # attn_fwd
        if bl:
            comp.append(bl["mlp_dgrad"]["id"])  # mlp_bwd (dgrad)
            comp.append(bl["mlp_wgrad"]["id"])  # mlp_bwd (wgrad)
        if fl:
            comp.append(fl["mlp"]["id"])  # mlp_fwd (expert)
        if bl:
            comp.append(bl["attn_dgrad"]["id"])  # attn_bwd (dgrad)
            comp.append(bl["attn_wgrad"]["id"])  # attn_bwd (wgrad)
        if bl and bl["combine_bwd"]:
            comm.append(bl["combine_bwd"]["id"])  # combine_bwd
        if fl and fl["dispatch"]:
            comm.append(fl["dispatch"]["id"])  # dispatch_fwd
        if bl and bl["dispatch_bwd"]:
            comm.append(bl["dispatch_bwd"]["id"])  # dispatch_bwd
        if fl and fl["combine"]:
            comm.append(fl["combine"]["id"])  # combine_fwd
    if F["post"]:
        comp.append(F["post"]["id"])  # lm-head fwd
    if B["post"]:
        comp.append(B["post"]["id"])  # embedding bwd


def schedule_overlap(layout, dense_set, t, m):
    """Two-stream (comp/comm) combined-1F1B schedule with EP A2A overlap.
    Returns (nodes, makespan, comp_busy[d], comm_busy[d]); each node gets
    'start'/'end' (ms). Streams run in issue order; a node starts at
    max(stream-free, all data-preds done) — the exact per-stream list schedule."""
    p, v = layout.pp, layout.vpp
    V = p * v
    g = _NodeGen()
    fwd = {
        (mb, vs): _fwd_op(g, mb, vs, layout, dense_set, t)
        for vs in range(V)
        for mb in range(m)
    }
    bwd = {
        (mb, vs): _bwd_op(g, mb, vs, layout, dense_set, t)
        for vs in range(V)
        for mb in range(m)
    }
    # cross-op / cross-device data deps (activation fwd, grad bwd, fwd-before-bwd)
    for (mb, vs), op in fwd.items():
        if vs > 0:
            op["first"]["preds"].append(fwd[(mb, vs - 1)]["last"]["id"])
    for (mb, vs), op in bwd.items():
        op["first"]["preds"].append(fwd[(mb, vs)]["last"]["id"])
        if vs < V - 1:
            op["first"]["preds"].append(bwd[(mb, vs + 1)]["last"]["id"])
    # per-device stream issue queues, following the 1F1B program (pair steady F+B)
    comp_q = {d: [] for d in range(p)}
    comm_q = {d: [] for d in range(p)}
    for d in range(p):
        prog = _pp_program(p, v, m, d)
        i = 0
        while i < len(prog):
            kind, c, mb, ph = prog[i]
            vs = c * p + d
            if (
                kind == "F"
                and ph == "steady"
                and i + 1 < len(prog)
                and prog[i + 1][0] == "B"
                and prog[i + 1][2] != mb
            ):
                # combine only INDEPENDENT microbatches (real combined-1F1B); a
                # same-mb F/B pair (e.g. the top stage's vs) is data-dependent —
                # B needs this F's output — so overlapping would deadlock. Issue
                # such a pair sequentially instead.
                bk, bc, bmb, bph = prog[i + 1]
                _issue_combined(
                    comp_q[d], comm_q[d], fwd[(mb, vs)], bwd[(bmb, bc * p + d)]
                )
                i += 2
            else:
                if kind == "F":
                    _issue_fwd(comp_q[d], comm_q[d], fwd[(mb, vs)])
                else:
                    _issue_bwd(comp_q[d], comm_q[d], bwd[(mb, vs)])
                i += 1
    # list-schedule: two in-order streams per device, cross-stream event deps
    by_id = {n["id"]: n for n in g.nodes}
    streams = []
    for d in range(p):
        streams.append(["comp", d, comp_q[d], 0, 0.0])
        streams.append(["comm", d, comm_q[d], 0, 0.0])
    end = {}
    remaining = sum(len(s[2]) for s in streams)
    while remaining > 0:
        progressed = False
        for s in streams:
            _, d, q, ptr, free = s
            if ptr >= len(q):
                continue
            n = by_id[q[ptr]]
            if all(pid in end for pid in n["preds"]):
                st = max(free, max((end[pid] for pid in n["preds"]), default=0.0))
                n["start"], n["end"] = st, st + n["ms"]
                end[n["id"]] = n["end"]
                s[3] += 1
                s[4] = n["end"]
                remaining -= 1
                progressed = True
        if not progressed:
            break
    makespan = max((n.get("end", 0.0) for n in g.nodes), default=0.0)
    comp_busy = [0.0] * p
    comm_busy = [0.0] * p
    for n in g.nodes:
        (comp_busy if n["stream"] == "comp" else comm_busy)[n["dev"]] += n["ms"]
    return g.nodes, makespan, comp_busy, comm_busy


# --- SVG rendering ------------------------------------------------------------
def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _color(tok):
    """(fill, stroke) for a phase token. Hue = family; lightness = fwd/dgrad/wgrad."""
    fam, hue = FAMILY[tok]
    sat = 22 if fam == "embed" else 62
    if tok in FWD_TOKENS:
        lig = 70
    elif tok.endswith("^W"):
        lig = 44
    elif tok.endswith("^D"):
        lig = 56
    else:  # D^, C^, V^, L^ (single backward box)
        lig = 52
    return (f"hsl({hue},{sat}%,{lig}%)", f"hsl({hue},{sat}%,{max(22, lig - 30)}%)")


def render_svg(ops, makespan, busy, layout, m, t, unit_ms, unit_name, px_per_unit, out):
    """All widths/labels are in UNITS (1 unit = unit_ms ms, the --unit phase).
    Every sub-phase box carries its value in units; each pp row shows its bubble%."""
    p, v = layout.pp, layout.vpp
    ppu = px_per_unit / unit_ms  # px per ms (derived)
    PAD, LBL = 26, 96
    ROW_H, ROW_GAP = 46, 16
    row_pitch = ROW_H + ROW_GAP
    top = 100  # space for title + axis
    x0 = PAD + LBL
    plot_w = makespan * ppu
    width = x0 + plot_w + PAD
    grid_bot = top + p * row_pitch - ROW_GAP

    ideal = max(busy) if busy else 0.0
    bubble = (makespan - ideal) / makespan if makespan else 0.0
    mk_u = makespan / unit_ms  # makespan in units

    def u(ms):  # ms -> units
        return ms / unit_ms

    # legend = a single horizontal flow (left->right), packed by label width and
    # wrapping only if the row would overrun the plot width
    toks = FWD_TOKENS + BWD_TOKENS

    def _leg_item_w(tok):
        txt = f"{LABEL[tok]} ({u(t[tok]):.2f}u)"
        return 13 + 6 + len(txt) * 6.0 + 22  # swatch + gap + text + trailing

    leg_wrap = max(plot_w + 6, 600.0)
    leg_pos, lx, row = [], 0.0, 0
    for tok in toks:
        iw = _leg_item_w(tok)
        if lx > 0 and lx + iw > leg_wrap:
            row += 1
            lx = 0.0
        leg_pos.append((lx, row, tok))
        lx += iw
    leg_rows = row + 1
    height = grid_bot + 34 + leg_rows * 20 + PAD

    b = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
        f'height="{height:.0f}" font-family="-apple-system,Segoe UI,Roboto,sans-serif">',
        f'<rect width="{width:.0f}" height="{height:.0f}" fill="#fafbfc"/>',
        f'<text x="{PAD}" y="32" font-size="17" font-weight="700" fill="#1b2733">'
        f"1F1B pipeline timeline &#8212; pp={p}, vpp={v}, {m} microbatches</text>",
        f'<text x="{PAD}" y="54" font-size="12.5" fill="#4b5563">'
        f"unit 1 = {_esc(unit_name)} = {unit_ms:.3f} ms &#183; box width &#8733; time "
        f"(per-phase units in the legend below) &#183; makespan {mk_u:.1f} u "
        f"({makespan:.0f} ms) &#183; busiest {u(ideal):.1f} u &#183; "
        f"overall bubble {bubble * 100:.1f}%</text>",
    ]

    # time grid + axis ticks, in UNITS (~12 ticks)
    raw = mk_u / 12 if mk_u else 1
    mag = 10 ** (len(str(int(raw))) - 1) if raw >= 1 else 1
    step = max(mag, round(raw / mag) * mag) if raw else 1
    tick = 0.0
    while tick <= mk_u + 1e-6:
        x = x0 + tick * px_per_unit
        b.append(
            f'<line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{grid_bot}" '
            f'stroke="#e7ebef" stroke-width="1"/>'
        )
        b.append(
            f'<text x="{x:.1f}" y="{top - 10}" font-size="10" fill="#9aa4b0" '
            f'text-anchor="middle">{tick:.0f}u</text>'
        )
        tick += step

    # per-stage (device) rows + PER-RANK BUBBLE
    for d in range(p):
        ry = top + d * row_pitch
        b.append(
            f'<rect x="{x0 - 3}" y="{ry:.0f}" width="{plot_w + 6:.1f}" '
            f'height="{ROW_H}" rx="5" fill="#fff" stroke="#cfd6de" '
            f'stroke-width="1.2"/>'
        )
        row_bubble = (makespan - busy[d]) / makespan if makespan else 0.0
        b.append(
            f'<text x="{PAD}" y="{ry + 15:.0f}" font-size="12.5" '
            f'font-weight="700" fill="#1b2733">pp {d}</text>'
        )
        b.append(
            f'<text x="{PAD}" y="{ry + 29:.0f}" font-size="10.5" '
            f'font-weight="700" fill="#b4530a">bubble {row_bubble * 100:.1f}%</text>'
        )
        b.append(
            f'<text x="{PAD}" y="{ry + 41:.0f}" font-size="8" fill="#8a95a1">'
            f"busy {u(busy[d]):.0f}u</text>"
        )

    # op boxes (segments); every box labels its value in units (rotated to fit),
    # vpp shown by a thin top stripe.
    seg_h = ROW_H - 12
    for o in ops:
        ry = top + o["stage"] * row_pitch
        x = x0 + o["t"] * ppu
        segs = (SEG_F if o["kind"] == "F" else SEG_B)[o["vs"]]
        stripe = "#94a3b8" if o["chunk"] == 0 else "#334155"
        b.append(
            f'<rect x="{x:.1f}" y="{ry + 3:.1f}" '
            f'width="{o["dur"] * ppu:.1f}" height="2.4" fill="{stripe}"/>'
        )
        cx = x
        for tok, ms in segs:
            w = ms * ppu
            val = u(ms)
            fill, stroke = _color(tok)
            b.append(
                f'<rect x="{cx:.2f}" y="{ry + 7:.1f}" width="{max(w, 0.4):.2f}" '
                f'height="{seg_h}" fill="{fill}" stroke="{stroke}" '
                f'stroke-width="0.5"><title>mb {o["mb"]} '
                f"{'F' if o['kind'] == 'F' else 'B'} vpp{o['chunk']} pp{o['stage']} "
                f"&#183; {_esc(tok)} = {val:.2f} u ({ms:.3f} ms)</title></rect>"
            )
            # phase glyph only (NO time in the box); value lives in the tooltip
            # + the bottom legend
            if w >= 8:
                g = GLYPH.get(tok, tok)
                b.append(
                    f'<text x="{cx + w / 2:.2f}" y="{ry + 7 + seg_h / 2 + 3.3:.1f}" '
                    f'font-size="9" fill="#0f1b28" text-anchor="middle">'
                    f"{_esc(g)}</text>"
                )
            cx += w
        if o["dur"] * ppu >= 12:  # microbatch id above the op
            b.append(
                f'<text x="{x + o["dur"] * ppu / 2:.1f}" y="{ry + 4:.1f}" '
                f'font-size="7.5" fill="#64748b" text-anchor="middle">'
                f"{o['mb']}</text>"
            )

    # legend — horizontal flow (positions computed above)
    ly = grid_bot + 34
    b.append(
        f'<text x="{PAD}" y="{ly - 13}" font-size="11" font-weight="700" '
        f'fill="#4b5563">phases (value in units) &#183; vpp stripe: '
        f'<tspan fill="#94a3b8">chunk0</tspan> / '
        f'<tspan fill="#334155">chunk1</tspan></text>'
    )
    for lxoff, r, tok in leg_pos:
        x = PAD + lxoff
        yy = ly + r * 20
        fill, stroke = _color(tok)
        b.append(
            f'<rect x="{x:.1f}" y="{yy - 10}" width="13" height="13" rx="2" '
            f'fill="{fill}" stroke="{stroke}"/>'
        )
        b.append(
            f'<text x="{x + 18:.1f}" y="{yy}" font-size="10.5" fill="#374151">'
            f"{_esc(LABEL[tok])} ({u(t[tok]):.2f}u)</text>"
        )

    b.append("</svg>")
    outp = out if out.endswith(".svg") else out + ".svg"
    Path(outp).write_text("".join(b))
    return outp, makespan, ideal, bubble


def _legend_flow(t, u, plot_w):
    """Shared horizontal legend layout -> (leg_pos, leg_rows)."""
    toks = FWD_TOKENS + BWD_TOKENS

    def iw(tok):
        return 13 + 6 + len(f"{LABEL[tok]} ({u(t[tok]):.2f}u)") * 6.0 + 22

    wrap = max(plot_w + 6, 600.0)
    pos, lx, row = [], 0.0, 0
    for tok in toks:
        w = iw(tok)
        if lx > 0 and lx + w > wrap:
            row += 1
            lx = 0.0
        pos.append((lx, row, tok))
        lx += w
    return pos, row + 1


def _draw_legend(b, leg_pos, t, u, PAD, ly):
    b.append(
        f'<text x="{PAD}" y="{ly - 13}" font-size="11" font-weight="700" '
        f'fill="#4b5563">phases (value in units) &#183; vpp stripe: '
        f'<tspan fill="#94a3b8">chunk0</tspan> / '
        f'<tspan fill="#334155">chunk1</tspan></text>'
    )
    for lxoff, r, tok in leg_pos:
        x, yy = PAD + lxoff, ly + r * 20
        fill, stroke = _color(tok)
        b.append(
            f'<rect x="{x:.1f}" y="{yy - 10}" width="13" height="13" rx="2" '
            f'fill="{fill}" stroke="{stroke}"/>'
        )
        b.append(
            f'<text x="{x + 18:.1f}" y="{yy}" font-size="10.5" fill="#374151">'
            f"{_esc(LABEL[tok])} ({u(t[tok]):.2f}u)</text>"
        )


def render_overlap_svg(
    nodes,
    makespan,
    comp_busy,
    comm_busy,
    layout,
    m,
    t,
    unit_ms,
    unit_name,
    px_per_unit,
    out,
):
    """EP-overlap timeline: two lanes per pp stage — compute (top) + comm (bottom)
    — so hidden A2A shows as a comm box sitting under a compute box; exposed A2A
    shows as a comm box over a compute gap. Bubble is measured on the comp lane."""
    p, v = layout.pp, layout.vpp
    ppu = px_per_unit / unit_ms
    PAD, LBL = 26, 112
    LANE_H, LANE_GAP, ROW_GAP = 22, 3, 20
    ROW_H = LANE_H * 2 + LANE_GAP
    row_pitch = ROW_H + ROW_GAP
    top = 104
    x0 = PAD + LBL
    plot_w = makespan * ppu
    width = x0 + plot_w + PAD
    grid_bot = top + p * row_pitch - ROW_GAP

    def u(ms):
        return ms / unit_ms

    leg_pos, leg_rows = _legend_flow(t, u, plot_w)
    height = grid_bot + 34 + leg_rows * 20 + PAD

    ideal = max(comp_busy) if comp_busy else 0.0
    bubble = (makespan - ideal) / makespan if makespan else 0.0
    mk_u = makespan / unit_ms

    b = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
        f'height="{height:.0f}" font-family="-apple-system,Segoe UI,Roboto,sans-serif">',
        f'<rect width="{width:.0f}" height="{height:.0f}" fill="#fafbfc"/>',
        f'<text x="{PAD}" y="32" font-size="17" font-weight="700" fill="#1b2733">'
        f"1F1B + EP-overlap timeline &#8212; pp={p}, vpp={v}, {m} microbatches</text>",
        f'<text x="{PAD}" y="54" font-size="12.5" fill="#4b5563">'
        f"unit 1 = {_esc(unit_name)} = {unit_ms:.3f} ms &#183; each stage has two "
        f'streams: <tspan font-weight="700">comp</tspan> (attn/mlp) over '
        f'<tspan font-weight="700">comm</tspan> (dispatch/combine A2A) &#183; A2A under '
        f"a compute box = hidden, over a gap = exposed &#183; makespan {mk_u:.1f} u "
        f"({makespan:.0f} ms) &#183; overall bubble {bubble * 100:.1f}% (comp lane)</text>",
    ]

    # unit ticks
    raw = mk_u / 12 if mk_u else 1
    mag = 10 ** (len(str(int(raw))) - 1) if raw >= 1 else 1
    step = max(mag, round(raw / mag) * mag) if raw else 1
    tick = 0.0
    while tick <= mk_u + 1e-6:
        x = x0 + tick * px_per_unit
        b.append(
            f'<line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{grid_bot}" '
            f'stroke="#e7ebef" stroke-width="1"/>'
        )
        b.append(
            f'<text x="{x:.1f}" y="{top - 10}" font-size="10" fill="#9aa4b0" '
            f'text-anchor="middle">{tick:.0f}u</text>'
        )
        tick += step

    # per-stage rows: comp lane (top) + comm lane (bottom), with per-rank bubble
    for d in range(p):
        ry = top + d * row_pitch
        for lane, ly0, tint in (
            ("comp", ry, "#ffffff"),
            ("comm", ry + LANE_H + LANE_GAP, "#f4f6f9"),
        ):
            b.append(
                f'<rect x="{x0 - 3}" y="{ly0:.0f}" width="{plot_w + 6:.1f}" '
                f'height="{LANE_H}" rx="4" fill="{tint}" stroke="#cfd6de" '
                f'stroke-width="1"/>'
            )
            b.append(
                f'<text x="{x0 - 8}" y="{ly0 + LANE_H / 2 + 3:.0f}" font-size="8" '
                f'fill="#9aa4b0" text-anchor="end">{lane}</text>'
            )
        rb = (makespan - comp_busy[d]) / makespan if makespan else 0.0
        b.append(
            f'<text x="{PAD}" y="{ry + 14:.0f}" font-size="12.5" '
            f'font-weight="700" fill="#1b2733">pp {d}</text>'
        )
        b.append(
            f'<text x="{PAD}" y="{ry + 28:.0f}" font-size="10.5" '
            f'font-weight="700" fill="#b4530a">bubble {rb * 100:.1f}%</text>'
        )
        b.append(
            f'<text x="{PAD}" y="{ry + 40:.0f}" font-size="8" fill="#8a95a1">'
            f"comp {u(comp_busy[d]):.0f}u comm {u(comm_busy[d]):.0f}u</text>"
        )

    # nodes
    for n in nodes:
        if "start" not in n:
            continue
        ry = top + n["dev"] * row_pitch
        ly0 = ry if n["stream"] == "comp" else ry + LANE_H + LANE_GAP
        x = x0 + n["start"] * ppu
        w = n["ms"] * ppu
        tok = n["tok"]
        fill, stroke = _color(tok)
        b.append(
            f'<rect x="{x:.2f}" y="{ly0 + 2:.1f}" width="{max(w, 0.4):.2f}" '
            f'height="{LANE_H - 4}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="0.5"><title>mb {n["mb"]} '
            f"{'F' if n['kind'] == 'F' else 'B'} vs{n['vs']} pp{n['dev']} &#183; "
            f"{_esc(tok)} = {u(n['ms']):.2f} u ({n['ms']:.3f} ms)</title></rect>"
        )
        if w >= 8:
            b.append(
                f'<text x="{x + w / 2:.2f}" y="{ly0 + LANE_H / 2 + 3.2:.1f}" '
                f'font-size="9" fill="#0f1b28" text-anchor="middle">'
                f"{_esc(GLYPH.get(tok, tok))}</text>"
            )

    _draw_legend(b, leg_pos, t, u, PAD, grid_bot + 34)
    b.append("</svg>")
    outp = out if out.endswith(".svg") else out + ".svg"
    Path(outp).write_text("".join(b))
    return outp, makespan, ideal, bubble


# module-level handles used by render_svg (set in main)
SEG_F, SEG_B = {}, {}


def parse_args():
    ap = argparse.ArgumentParser(
        description="Op-level wall-clock 1F1B pipeline timeline SVG (interleaved/vpp), "
        "each op decomposed into measured GPU sub-phases (F/D/E/C/M/V/L + grads)."
    )
    ap.add_argument("--pp", type=int, required=True, help="pipeline-parallel size")
    ap.add_argument(
        "--pipeline-model-parallel-layout",
        "--pp-layout",
        dest="layout",
        required=True,
        metavar="DSL",
        help='Megatron layout, e.g. "Ett|tttttt|...|tttL" (vpp inferred)',
    )
    ap.add_argument(
        "--microbatches",
        "--mbs",
        dest="m",
        type=int,
        default=32,
        help="number of microbatches in the pipeline (default 32)",
    )
    ap.add_argument(
        "--dense-layers",
        default="0",
        metavar="IDS",
        help="comma list of global transformer-layer ids that are DENSE "
        '(F,M forward); default "0". "none" = all MoE',
    )
    ap.add_argument(
        "--config",
        default=None,
        metavar="times.json",
        help='JSON of per-phase times {token: ms}, e.g. {"F":1.4,"E":1.59,"E^D":1.9}; '
        "keys are phase tokens (see legend). Overrides built-in defaults; "
        "--t-<phase> flags override this in turn.",
    )
    ap.add_argument(
        "--dump-config",
        action="store_true",
        help="print the effective per-phase times as a --config JSON and exit",
    )
    ap.add_argument(
        "--unit",
        default="F",
        metavar="PHASE",
        help="phase token whose time = 1 unit (default F = attn_fwd); "
        "every box value is time/unit",
    )
    ap.add_argument(
        "--unit-ms",
        type=float,
        default=None,
        metavar="MS",
        help="explicit ms for 1 unit (overrides --unit)",
    )
    ap.add_argument(
        "--px-per-unit",
        type=float,
        default=13.0,
        metavar="PX",
        help="horizontal scale (default 13 px per unit)",
    )
    ap.add_argument(
        "--ep-overlap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="model MoE EP A2A overlap (Megatron combined-1F1B): split "
        "each stage into compute + comm streams and hide dispatch/"
        "combine A2A behind the paired microbatch's compute "
        "(default on; --no-ep-overlap for the serial single-lane view)",
    )
    ap.add_argument("--svg", metavar="OUT", help="output .svg path")
    # per-phase time overrides (ms); default from --config, else built-in DEFAULTS
    g = ap.add_argument_group("per-phase time overrides (ms; > --config > default)")
    for tok in FWD_TOKENS + BWD_TOKENS:
        g.add_argument(
            f"--t-{_flag(tok)}",
            type=float,
            default=None,
            dest="t_" + _flag(tok),
            metavar="MS",
            help=f"{LABEL[tok]}",
        )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    t = resolve_times(args, config)
    if args.dump_config:
        print(
            json.dumps(
                {tok: round(t[tok], 4) for tok in FWD_TOKENS + BWD_TOKENS}, indent=2
            )
        )
        sys.exit(0)
    if not args.svg:
        sys.exit("--svg OUT is required (or pass --dump-config to just print times)")
    layout = PPLayout(args.pp, dsl=args.layout)

    if args.dense_layers.strip().lower() in ("", "none"):
        dense_set = set()
    else:
        dense_set = {int(x) for x in args.dense_layers.split(",") if x.strip() != ""}

    SEG_F, SEG_B = build_durations(layout, dense_set, t)
    dur_f = {vs: sum(ms for _, ms in segs) for vs, segs in SEG_F.items()}
    dur_b = {vs: sum(ms for _, ms in segs) for vs, segs in SEG_B.items()}

    # unit anchor: 1 unit = --unit-ms, else the --unit phase's time
    if args.unit_ms is not None:
        unit_ms, unit_name = args.unit_ms, f"{args.unit_ms:.3f}ms"
    else:
        if args.unit not in t:
            sys.exit(
                f"--unit {args.unit!r} not a phase token; pick from {FWD_TOKENS + BWD_TOKENS}"
            )
        unit_ms = t[args.unit]
        unit_name = f"{args.unit} ({SRC.get(args.unit, '-')})"
    if unit_ms <= 0:
        sys.exit(f"unit time is {unit_ms} ms; pick a non-zero --unit / --unit-ms")

    # text summary
    print("=" * 74)
    print(
        f"pp={layout.pp}  vpp={layout.vpp}  layers={layout.num_layers}  "
        f"microbatches={args.m}  dense={sorted(dense_set) or 'none'}"
    )
    print(f"unit 1 = {unit_name} = {unit_ms:.4f} ms")
    print("per-phase   ms      units   source")
    for tok in FWD_TOKENS + BWD_TOKENS:
        src = (
            "cli"
            if getattr(args, "t_" + _flag(tok)) is not None
            else ("config" if tok in config else "default")
        )
        print(
            f"    {tok:<4} {t[tok]:7.3f}  {t[tok] / unit_ms:6.2f}u  ({src}: {SRC.get(tok, '-')})"
        )
    print("per pp-stage chunk durations (forward | backward, units):")
    for d in range(layout.pp):
        parts = []
        for c in range(layout.vpp):
            vs = c * layout.pp + d
            parts.append(
                f"vpp{c} F={dur_f[vs] / unit_ms:.1f}u B={dur_b[vs] / unit_ms:.1f}u"
            )
        print(f"    pp {d}: " + "   ".join(parts))

    if args.ep_overlap:
        nodes, makespan, comp_busy, comm_busy = schedule_overlap(
            layout, dense_set, t, args.m
        )
        outp, mk, ideal, bubble = render_overlap_svg(
            nodes,
            makespan,
            comp_busy,
            comm_busy,
            layout,
            args.m,
            t,
            unit_ms,
            unit_name,
            args.px_per_unit,
            args.svg,
        )
        # serial reference (no overlap) to quantify the A2A hiding benefit
        _, ser_mk, _ = schedule_timed(layout.pp, layout.vpp, args.m, dur_f, dur_b)
        print("-" * 74)
        print(
            f"[EP-overlap]  makespan {mk / unit_ms:.1f} u ({mk:.0f} ms)   "
            f"busiest comp lane {ideal / unit_ms:.1f} u   overall bubble "
            f"{bubble * 100:.1f}%"
        )
        print(
            f"serial (no overlap) makespan {ser_mk / unit_ms:.1f} u ({ser_mk:.0f} ms)"
            f"  ->  overlap saves {(1 - mk / ser_mk) * 100:.1f}%"
            if ser_mk
            else ""
        )
        print(
            "per-rank bubble (comp lane):  "
            + "  ".join(
                f"pp{d}={(mk - comp_busy[d]) / mk * 100:.1f}%" for d in range(layout.pp)
            )
        )
        print(
            "per-rank comm/comp (units):   "
            + "  ".join(
                f"pp{d}={comm_busy[d] / unit_ms:.0f}/{comp_busy[d] / unit_ms:.0f}"
                for d in range(layout.pp)
            )
        )
        print(f"Wrote EP-overlap timeline SVG -> {outp}")
    else:
        ops, makespan, buspy = schedule_timed(
            layout.pp, layout.vpp, args.m, dur_f, dur_b
        )
        outp, mk, ideal, bubble = render_svg(
            ops,
            makespan,
            buspy,
            layout,
            args.m,
            t,
            unit_ms,
            unit_name,
            args.px_per_unit,
            args.svg,
        )
        print("-" * 74)
        print(
            f"makespan  {mk / unit_ms:.1f} u ({mk:.0f} ms)   busiest-device "
            f"{ideal / unit_ms:.1f} u   overall bubble {bubble * 100:.1f}%"
        )
        print(
            "per-rank bubble:  "
            + "  ".join(
                f"pp{d}={(mk - buspy[d]) / mk * 100:.1f}%" for d in range(layout.pp)
            )
        )
        print(f"Wrote timeline SVG -> {outp}")
