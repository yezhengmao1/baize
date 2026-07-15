"""
gpu-groups: resolve the per-rank parallel process groups from a parallelism
config, the way Megatron-LM lays them out — so you can tell, for any rank, which
ranks it talks to under each kind of communication (TP / SP / CP / DP / PP / EP).

This is a pure-config tool (no profile needed). It reimplements Megatron's
``RankGenerator`` / ``generate_masked_orthogonal_rank_groups``
(megatron/core/parallel_state.py): the global rank is laid out as a mixed-radix
number over the parallel axes in ``--order`` (default tp-sp-cp-ep-dp-pp), and
each communication group is the set of ranks that differ only along that axis.

Two generators, exactly as Megatron:
  * decoder / attention : axes tp, sp, cp, dp, pp   (ep = 1)   -> TP/SP/CP/DP/PP
  * expert / MoE        : axes tp(=etp), sp, ep, dp(=edp), pp (cp = 1) -> EP/ETP/EDP
    where  dp  = world / (tp * sp * cp * pp)
           edp = world / (etp * sp * ep * pp)   (cp folds into the expert DP)

SP (sequence parallel) is configurable via ``--sp``:
  * --sp 1 (default) : Megatron semantics — SP rides the TP group.
  * --sp N>1         : SP is its own axis that further splits the ranks.

Pipeline layer layout (--pipeline-model-parallel-layout / --pp-layout) reuses
Megatron's PipelineParallelLayerLayout DSL (E=embedding, t=decoder layer,
L=loss, m=MTP; `|` = stage, `x*n` and `(...)*n` unroll). Virtual pipeline is
inferred (vpp = stages // pp); it does not change the rank groups, only which
layers each pp rank holds (interleaved, vpp-major).

Author: yezhengmaolove@gmail.com
"""

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import List


# --- Megatron rank math (ported verbatim; general over any axis layout) -------
def generate_masked_orthogonal_rank_groups(
    world_size: int, parallel_size: List[int], mask: List[bool]
) -> List[List[int]]:
    """Megatron's group generator: for the axes flagged True in ``mask``, return
    every group of ranks that vary only along those axes (orthogonal to the rest)."""

    def prefix_product(a, init=1):
        r = [init]
        for v in a:
            init = init * v
            r.append(init)
        return r

    def inner_product(a, b):
        return sum(x * y for x, y in zip(a, b))

    def decompose(index, shape, stride=None):
        if stride is None:
            stride = prefix_product(shape)
        idx = [(index // d) % s for s, d in zip(shape, stride)]
        assert sum(x * y for x, y in zip(idx, stride[:-1])) == index
        return idx

    masked_shape = [s for s, m in zip(parallel_size, mask) if m]
    unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]

    global_stride = prefix_product(parallel_size)
    masked_stride = [d for d, m in zip(global_stride, mask) if m]
    unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]

    group_size = prefix_product(masked_shape)[-1]
    num_of_group = world_size // group_size

    ranks = []
    for group_index in range(num_of_group):
        decomposed_group_idx = decompose(group_index, unmasked_shape)
        rank = []
        for rank_in_group in range(group_size):
            decomposed_rank_idx = decompose(rank_in_group, masked_shape)
            rank.append(
                inner_product(decomposed_rank_idx, masked_stride)
                + inner_product(decomposed_group_idx, unmasked_stride)
            )
        ranks.append(rank)
    return ranks


class RankGenerator:
    """Megatron RankGenerator, generalized to an arbitrary ordered axis map.

    ``sizes`` maps axis name -> size; ``order`` is the axis order (mixed-radix,
    left = fastest-varying / innermost). Axes with size 1 may be omitted from
    ``order``; they are appended so the world size still matches."""

    def __init__(self, sizes: dict, order: List[str]):
        self.sizes = dict(sizes)
        order = [t for t in order if t in self.sizes]
        for name, s in self.sizes.items():
            if name not in order:
                if s != 1:
                    raise ValueError(f"axis '{name}' has size {s} but is not in order {order}")
                order.append(name)
        self.order = order
        self.ordered_size = [self.sizes[t] for t in order]
        self.world_size = 1
        for s in self.ordered_size:
            self.world_size *= s

    def get_ranks(self, token: str) -> List[List[int]]:
        toks = token.split("-")
        mask = [t in toks for t in self.order]
        return generate_masked_orthogonal_rank_groups(
            self.world_size, self.ordered_size, mask
        )


# --- pipeline layer layout (Megatron PipelineParallelLayerLayout DSL) ---------
LAYER_NAME = {"E": "embed", "t": "layer", "L": "loss", "m": "mtp"}


def _compress(ids: List[int]) -> str:
    """[0,1,20,21,22,23] -> '0-1,20-23'."""
    if not ids:
        return "-"
    ids = sorted(ids)
    out, lo, prev = [], ids[0], ids[0]
    for x in ids[1:]:
        if x == prev + 1:
            prev = x
            continue
        out.append(f"{lo}-{prev}" if lo != prev else f"{lo}")
        lo = prev = x
    out.append(f"{lo}-{prev}" if lo != prev else f"{lo}")
    return ",".join(out)


def parse_layout_dsl(s: str) -> List[List[str]]:
    """Megatron parse_str_to_list: unroll (...)*n then x*n, split on '|'."""
    s = s.replace(",", "")
    s = re.sub(r"\(([^)]+)\)\*(\d+)", lambda m: m.group(1) * int(m.group(2)), s)
    s = re.sub(r"(.)\*(\d+)", lambda m: m.group(1) * int(m.group(2)), s)
    stages = [list(seg) for seg in s.split("|")]
    for st in stages:
        for c in st:
            if c not in LAYER_NAME:
                raise SystemExit(
                    f"unknown layout char '{c}'; known: {list(LAYER_NAME)}"
                )
    return stages


class PPLayout:
    """Per-pp-rank layer assignment, with virtual-pipeline interleaving.

    Global decoder-layer ids are assigned in written order (vpp-major, pp-minor),
    matching Megatron get_layer_offset: chunk i -> pp_rank i%pp, vpp_rank i//pp."""

    def __init__(self, pp: int, dsl: str = None, simple: List[int] = None):
        flat = parse_layout_dsl(dsl) if dsl is not None else [["t"] * n for n in simple]
        if len(flat) % pp != 0:
            raise SystemExit(f"layout has {len(flat)} stages, not a multiple of pp={pp}")
        self.pp = pp
        self.vpp = len(flat) // pp
        nid = 0
        stage_layers = []
        for st in flat:
            ids = [nid + i for i in range(st.count("t"))]
            nid += len(ids)
            stage_layers.append(ids)
        self.num_layers = nid
        self.by_pp = [[] for _ in range(pp)]
        for i, st in enumerate(flat):
            self.by_pp[i % pp].append(
                {"vpp": i // pp, "chars": st, "layers": stage_layers[i],
                 "E": "E" in st, "L": "L" in st, "m": "m" in st}
            )

    def layer_ids(self, pp_rank: int) -> List[int]:
        return [x for ch in self.by_pp[pp_rank] for x in ch["layers"]]

    def flags(self, pp_rank: int) -> str:
        chunks = self.by_pp[pp_rank]
        return "".join(f for f in "ELm" if any(c[f] for c in chunks))

    def emb_stages(self) -> set:
        return {s for s in range(self.pp) if "E" in self.flags(s) or "L" in self.flags(s)}


# --- config -------------------------------------------------------------------
class Config:
    def __init__(self, args):
        tp, sp, cp, pp, ep = args.tp, args.sp, args.cp, args.pp, args.ep
        etp = args.etp if args.etp is not None else tp
        model = tp * sp * cp * pp
        if args.world is not None:
            world = args.world
            if world % model != 0:
                raise SystemExit(f"world {world} not divisible by tp*sp*cp*pp = {model}")
            dp = world // model
            if args.dp is not None and args.dp != dp:
                raise SystemExit(f"--dp {args.dp} conflicts with derived dp {dp}")
        else:
            dp = args.dp if args.dp is not None else 1
            world = model * dp

        emodel = etp * sp * ep * pp
        if world % emodel != 0:
            raise SystemExit(f"world {world} not divisible by expert etp*sp*ep*pp = {emodel}")
        edp = world // emodel

        self.tp, self.sp, self.cp, self.pp, self.ep = tp, sp, cp, pp, ep
        self.etp, self.dp, self.edp, self.world = etp, dp, edp, world
        self.order = args.order

        self.dec = RankGenerator(
            {"tp": tp, "sp": sp, "cp": cp, "dp": dp, "pp": pp}, args.order.split("-")
        )
        self.exp = RankGenerator(
            {"tp": etp, "sp": sp, "ep": ep, "dp": edp, "pp": pp}, args.order.split("-")
        )
        assert self.dec.world_size == world == self.exp.world_size

        self.layout = None
        if args.pipeline_model_parallel_layout:
            self.layout = PPLayout(pp, dsl=args.pipeline_model_parallel_layout)
        elif args.pp_layout:
            self.layout = PPLayout(pp, simple=[int(x) for x in args.pp_layout.split(",")])


def comm_kinds(cfg: Config):
    """One communication kind = (label, generator, token, description)."""
    kinds = []
    tp_desc = "tensor-parallel all-reduce"
    if cfg.sp == 1:
        tp_desc += " (+ sequence-parallel all-gather/reduce-scatter)"
    kinds.append(("TP", cfg.dec, "tp", tp_desc))
    if cfg.sp > 1:
        kinds.append(("SP", cfg.dec, "sp", "sequence-parallel all-gather/reduce-scatter"))
    if cfg.cp > 1:
        kinds.append(("CP", cfg.dec, "cp", "context-parallel attention (ring / all-gather)"))
    kinds.append(("DP", cfg.dec, "dp-cp", "data-parallel gradient reduce / optimizer (incl. CP)"))
    kinds.append(("PP", cfg.dec, "pp", "pipeline send/recv activations & grads (neighbors)"))
    if cfg.ep > 1:
        kinds.append(("EP", cfg.exp, "ep", "MoE expert all-to-all (dispatch / combine)"))
        kinds.append(("EDP", cfg.exp, "dp", "expert data-parallel gradient reduce"))
        if cfg.etp > 1:
            kinds.append(("ETP", cfg.exp, "tp", "expert tensor-parallel all-reduce"))
    return kinds


def invert(groups: List[List[int]]):
    m = {}
    for gi, g in enumerate(groups):
        for r in g:
            m[r] = (gi, g)
    return m


def embedding_ranks(cfg: Config):
    """First/last pp-stage rank of every pp group (word-embedding sync); if a
    layout is given, the stages actually carrying E / L."""
    stages = cfg.layout.emb_stages() if cfg.layout else {0, cfg.pp - 1}
    emb = set()
    for pg in cfg.dec.get_ranks("pp"):
        for s in stages:
            emb.add(pg[s])
    return emb


# --- printing -----------------------------------------------------------------
def _stage_desc(cfg: Config, stage: int) -> str:
    chunks = cfg.layout.by_pp[stage]
    parts = []
    for c in chunks:
        fl = "".join(f for f in "ELm" if c[f])
        parts.append(f"vpp{c['vpp']}[L{_compress(c['layers'])}]" + (fl or ""))
    return "  ".join(parts)


def print_summary(cfg: Config):
    print("=" * 78)
    print(f"world = {cfg.world}   order = {cfg.order}")
    print(f"  tp={cfg.tp}  sp={cfg.sp}  cp={cfg.cp}  dp={cfg.dp}  pp={cfg.pp}   "
          f"(dp = world/(tp*sp*cp*pp))")
    if cfg.ep > 1:
        print(f"  expert: etp={cfg.etp}  ep={cfg.ep}  edp={cfg.edp}   "
              f"(edp = world/(etp*sp*ep*pp))")
    if cfg.sp == 1:
        print("  SP: rides the TP group (Megatron default; pass --sp N for a separate axis)")
    if cfg.layout:
        L = cfg.layout
        extra = " + MTP" if any("m" in L.flags(s) for s in range(cfg.pp)) else ""
        print(f"  pp layout: vpp={L.vpp}, {L.num_layers} decoder layers{extra}")
        for s in range(cfg.pp):
            print(f"    pp-stage {s}: {_stage_desc(cfg, s)}")
    print("=" * 78)


def print_group_tables(cfg: Config):
    for label, gen, token, desc in comm_kinds(cfg):
        groups = gen.get_ranks(token)
        print(f"\n[{label}] {desc}")
        print(f"  {len(groups)} group(s) of {len(groups[0])} rank(s)  (token '{token}')")
        for gi, g in enumerate(groups):
            print(f"    {label}#{gi:<3} {g}")


def rank_line(cfg: Config, r: int, maps, emb) -> str:
    parts = [f"rank {r:>4}"]
    pp_gi, pp_g = maps["PP"][r]
    stage = pp_g.index(r)
    seg = ""
    if cfg.layout is not None:
        fl = cfg.layout.flags(stage)
        seg = f"(L{_compress(cfg.layout.layer_ids(stage))}{'|' + fl if fl else ''})"
    parts.append(f"pp#{pp_gi} stage{stage}{seg}")
    for label in maps:
        if label == "PP":
            continue
        parts.append(f"{label}#{maps[label][r][0]}")
    if r in emb:
        parts.append("emb")
    return "  ".join(parts)


def print_per_rank(cfg: Config, maps, emb, limit):
    print("\nper-rank groups (IDs cross-reference the tables above):")
    n = cfg.world if limit is None else min(limit, cfg.world)
    for r in range(n):
        print("  " + rank_line(cfg, r, maps, emb))
    if n < cfg.world:
        print(f"  ... ({cfg.world - n} more; use --csv for all or --rank R for one)")


def print_rank_detail(cfg: Config, r: int, maps, emb):
    print("=" * 78)
    pp_g = maps["PP"][r][1]
    stage = pp_g.index(r)
    tail = ""
    if cfg.layout is not None:
        tail = f",  layers L{_compress(cfg.layout.layer_ids(stage))}   [{_stage_desc(cfg, stage)}]"
    print(f"rank {r}:  pp stage {stage}/{cfg.pp - 1}{tail}"
          + ("   [embedding rank]" if r in emb else ""))
    print("=" * 78)
    for label, _, _, desc in comm_kinds(cfg):
        gi, g = maps[label][r]
        peers = [x for x in g if x != r]
        print(f"  {label:<4} {desc}")
        print(f"       group #{gi}: {g}")
        if label == "PP":
            i = g.index(r)
            print(f"       recv<-prev: {g[i - 1] if i > 0 else None}   "
                  f"send->next: {g[i + 1] if i < len(g) - 1 else None}")
        else:
            print(f"       peers: {peers if peers else '(none — group of 1)'}")


def write_csv(cfg: Config, maps, emb, path):
    labels = list(maps.keys())
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        head = ["rank", "pp_stage", "layers", "flags", "embedding"]
        for label in labels:
            head += [f"{label}_gid", f"{label}_members"]
        w.writerow(head)
        for r in range(cfg.world):
            stage = maps["PP"][r][1].index(r)
            layers = _compress(cfg.layout.layer_ids(stage)) if cfg.layout else ""
            flags = cfg.layout.flags(stage) if cfg.layout else ""
            row = [r, stage, layers, flags, int(r in emb)]
            for label in labels:
                gi, g = maps[label][r]
                row += [gi, " ".join(map(str, g))]
            w.writerow(row)
    print(f"\nWrote per-rank CSV -> {path}")


# --- pipeline schedule (1F1B, vpp-interleaved) --------------------------------
def _schedule_table(m: int, v: int, g: int) -> list[tuple]:
    """Megatron get_schedule_table: (microbatch_id, model_chunk_id) indexed by
    virtual_microbatch_id. A group of ``g`` microbatches runs on chunk 0, then the
    same g on chunk 1, ... (the last group grabs all remaining microbatches)."""
    table: list[tuple] = []
    for lo in range(0, m, g):
        hi = m if lo + g >= m else lo + g
        for c in range(v):
            for mb in range(lo, hi):
                table.append((mb, c))
    return table


def _pp_program(
    p: int, v: int, m: int, d: int, g: int | None = None, overlap: bool = False
) -> list[tuple]:
    """Megatron's fixed 1F1B op order for pipeline stage ``d`` — a list of
    (kind, chunk, microbatch, phase). Faithful port of
    forward_backward_pipelining_with_interleaving: warmup forwards, steady 1F1B
    (forward vmb ``k+warmup`` paired with backward vmb ``k``), then cooldown
    backwards. (mb, chunk) come from ``_schedule_table``; backward reverses the
    chunk (``v-1-chunk``).

    ``g`` = ``--microbatch-group-size-per-virtual-pipeline-stage`` (how many
    microbatches run contiguously on one chunk before advancing). Default = ``p``.
    ``overlap`` (EP-A2A combined-1F1B) adds one extra warmup forward so each steady
    step pairs INDEPENDENT forward/backward microbatches.

    mcore constraint (schedules.py:1006, only when interleaved v>1): the group must
    satisfy ``p <= g <= m`` and ``m % g`` in ``{0} ∪ [p, ∞)``. ``g < p`` makes the
    interleaved schedule INFEASIBLE — an early backward on a late stage depends on a
    backward of an earlier stage whose own warmup forward depends back on that late
    stage's not-yet-issued forward (a true dependency cycle, unbreakable by P2P
    buffering), which is exactly why mcore rejects it."""
    if g is None:
        g = p
    if v > 1:  # microbatch grouping only exists when interleaved
        if not (p <= g <= m):
            raise ValueError(
                f"microbatch_group_size_per_vp_stage={g} must be in [pp={p}, m={m}] "
                f"(mcore constraint; g<pp deadlocks the interleaved schedule)"
            )
        if 0 < (m % g) < p:
            raise ValueError(
                f"m % g = {m % g} must be 0 or >= pp={p} (else dependency bubbles)"
            )
    total = m * v
    table = _schedule_table(m, v, g)
    mb_of = [e[0] for e in table]
    chunk_of = [e[1] for e in table]

    if v == 1:
        num_warmup = p - 1 - d
    else:
        num_warmup = (p - 1 - d) * 2 + (v - 1) * g
        if overlap:
            num_warmup += 1
    num_warmup = min(num_warmup, total)
    remaining = total - num_warmup

    prog: list[tuple] = []  # (kind, chunk, microbatch, phase)
    for k in range(num_warmup):  # warmup: forwards only
        prog.append(("F", chunk_of[k], mb_of[k], "warmup"))
    for k in range(remaining):  # steady: forward(k+warmup) paired with backward(k)
        fk = k + num_warmup
        prog.append(("F", chunk_of[fk], mb_of[fk], "steady"))
        prog.append(("B", v - 1 - chunk_of[k], mb_of[k], "steady"))
    for k in range(remaining, total):  # cooldown: backwards only
        prog.append(("B", v - 1 - chunk_of[k], mb_of[k], "cooldown"))
    return prog


def pp_schedule(p: int, v: int, m: int) -> tuple[list[dict], int]:
    """1F1B schedule for ONE pipeline group (p stages, v vpp chunks, m microbatches).

    Each device runs Megatron's fixed 1F1B op order (`_pp_program`) in program
    order; op timings come from the data dependencies — a microbatch flows forward
    through virtual stages vs = chunk*p + stage (0..p*v-1) and backward in reverse,
    each op taking one unit. Returns (ops, makespan) with each op
    {kind:'F'|'B', mb, chunk, stage, t} (t = integer start time)."""
    V = p * v
    programs = [_pp_program(p, v, m, d) for d in range(p)]
    ptr = [0] * p
    dev_t = [0] * p
    end: dict = {}          # (kind, mb, vs) -> completion time
    ops: list[dict] = []
    total_ops = sum(len(pr) for pr in programs)
    stalls = 0
    while len(ops) < total_ops:
        progressed = False
        for d in range(p):
            if ptr[d] >= len(programs[d]):
                continue
            kind, c, mb, phase = programs[d][ptr[d]]
            vs = c * p + d
            if kind == "F":  # needs the same microbatch out of the previous vs
                if vs > 0 and ("F", mb, vs - 1) not in end:
                    continue
                dep = end.get(("F", mb, vs - 1), 0) if vs > 0 else 0
            else:            # backward needs its own forward + the next vs's backward
                if ("F", mb, vs) not in end:
                    continue
                if vs < V - 1 and ("B", mb, vs + 1) not in end:
                    continue
                dep = max(
                    end.get(("B", mb, vs + 1), 0) if vs < V - 1 else 0,
                    end[("F", mb, vs)],
                )
            t = max(dev_t[d], dep)
            end[(kind, mb, vs)] = t + 1
            dev_t[d] = t + 1
            ops.append({"kind": kind, "mb": mb, "chunk": c, "stage": d,
                        "t": t, "phase": phase})
            ptr[d] += 1
            progressed = True
        if not progressed:  # safety: a correct 1F1B program never deadlocks
            stalls += 1
            if stalls > total_ops + 10:
                break
    makespan = max((o["t"] + 1 for o in ops), default=0)
    return ops, makespan


# --- SVG visualization: DP data-flow, organized by physical node --------------
# The world is laid out as physical nodes (one row per node, GPUS_PER_NODE GPUs
# each).  Each GPU is colored by its **data shard** — the data-parallel replica it
# belongs to (its position along the dp axis): same color == same input batch,
# different color == different data.  The overlaid links are the **DP gradient
# all-reduce**: the ranks that hold the same model shard but see *different* data
# and average their gradients — i.e. how data actually flows in the DP dimension.
#
# The static chrome (outer <svg>, background, CSS classes) lives in
# nsys_tools/templates/parallel_groups.svg — kept out of this module so editors
# highlight it as SVG; placeholders __WIDTH__ / __HEIGHT__ / __CONTENT__.
_SVG_CW, _SVG_CH, _SVG_GAP = 46, 34, 6        # GPU-cell width / height / gap between GPUs
_SVG_PAD, _SVG_LBL = 24, 66                    # page margin / left node-label column
_SVG_STAGE = 52                                # left gutter grouping node rows by pp stage
_SVG_NODE_GAP = 26                             # vertical gap between node rows (room for links)
_SVG_EPBAND = 18                               # band above each node row for EP all-to-all arcs

_SVG_TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent / "templates" / "parallel_groups.svg"
)
GROUPS_SVG_TEMPLATE = _SVG_TEMPLATE_PATH.read_text()


def _svg_color(i: int):
    """Distinct pastel fill + matching darker stroke per data shard (golden-angle hue)."""
    hue = (i * 137.508) % 360
    return f"hsl({hue:.0f} 70% 88%)", f"hsl({hue:.0f} 45% 42%)"


def _svg_esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def data_shards(cfg: Config):
    """Map each rank to its data shard (dp-axis position): same shard == same input
    batch.  Returns (shard_of: dict rank->shard, n_shards)."""
    dp_groups = cfg.dec.get_ranks("dp")
    shard_of = {r: pos for g in dp_groups for pos, r in enumerate(g)}
    return shard_of, (len(dp_groups[0]) if dp_groups else 1)


def pp_stages(cfg: Config):
    """Map each rank to its pipeline stage (its position along the pp axis)."""
    pp_groups = cfg.dec.get_ranks("pp")
    stage_of = {r: pos for g in pp_groups for pos, r in enumerate(g)}
    return stage_of, (len(pp_groups[0]) if pp_groups else 1)


def _svg_content(cfg: Config, gpn: int) -> tuple[str, int, int]:
    """Build the node-organized DP data-flow SVG and its (width, height)."""
    world = cfg.world
    step = _SVG_CW + _SVG_GAP
    nodes = (world + gpn - 1) // gpn
    x_cells = _SVG_PAD + _SVG_STAGE + _SVG_LBL
    node_w = gpn * step - _SVG_GAP
    width = x_cells + node_w + _SVG_PAD
    ep_band = _SVG_EPBAND if cfg.ep > 1 else 0
    node_pitch = ep_band + _SVG_CH + _SVG_NODE_GAP

    shard_of, n_shards = data_shards(cfg)
    stage_of, npp = pp_stages(cfg)
    ar_groups = cfg.dec.get_ranks("dp-cp")             # DP gradient all-reduce
    ep_groups = cfg.exp.get_ranks("ep") if cfg.ep > 1 else []   # expert all-to-all

    body: List[str] = []
    grid_top = _SVG_PAD

    def cell_top(n: int) -> float:
        return grid_top + n * node_pitch + ep_band     # cells sit below the EP band

    def geom(r: int):
        n, p = divmod(r, gpn)
        return n, x_cells + p * step + _SVG_CW / 2, cell_top(n)   # (row, x-center, y-top)

    # 1) pp-stage gutter: bracket every run of consecutive node rows on the same
    #    stage (pp is the outermost axis, so a stage is a contiguous block of nodes)
    node_stage = [stage_of.get(n * gpn, 0) for n in range(nodes)]
    n0 = 0
    while n0 < nodes:
        n1 = n0
        while n1 + 1 < nodes and node_stage[n1 + 1] == node_stage[n0]:
            n1 += 1
        top = grid_top + n0 * node_pitch - 5
        bot = cell_top(n1) + _SVG_CH + 5
        midy = (top + bot) / 2
        gx = _SVG_PAD
        body.append(
            f'<rect class="stagebox" x="{gx}" y="{top:.1f}" width="{_SVG_STAGE - 10}" '
            f'height="{bot - top:.1f}" rx="7"/>'
            f'<text class="stagelbl" x="{gx + (_SVG_STAGE - 10) / 2:.1f}" y="{midy:.1f}" '
            f'transform="rotate(-90 {gx + (_SVG_STAGE - 10) / 2:.1f} {midy:.1f})">'
            f'pp stage {node_stage[n0]}</text>'
        )
        n0 = n1 + 1

    # 2) node backgrounds + labels
    for n in range(nodes):
        ct = cell_top(n)
        body.append(
            f'<rect class="nodebox" x="{x_cells - 5}" y="{ct - 5:.1f}" '
            f'width="{node_w + 10}" height="{_SVG_CH + 10}" rx="7"/>'
            f'<text class="nodelbl" x="{_SVG_PAD + _SVG_STAGE}" '
            f'y="{ct + _SVG_CH / 2 + 4:.1f}">node {n}</text>'
        )

    # 3) EP expert all-to-all (MoE dispatch/combine): every GPU in an expert group
    #    exchanges tokens with all others — full mesh, arcs bulging up into the band
    #    above each node (a separate channel from the DP all-reduce below).
    for g in ep_groups:
        pts = [geom(r) for r in g]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                (ra, xa, ya), (rb, xb, yb) = pts[i], pts[j]
                if ra == rb:                    # same node row: arc up into the EP band
                    h = min(ep_band - 2, 6 + 0.10 * abs(xb - xa))
                    body.append(
                        f'<path class="ep" d="M{xa:.1f},{ya:.1f} '
                        f'Q{(xa + xb) / 2:.1f},{ya - h:.1f} {xb:.1f},{yb:.1f}"/>'
                    )
                else:                           # cross-node expert group: straight link
                    body.append(f'<path class="ep" d="M{xa:.1f},{ya:.1f} '
                                f'L{xb:.1f},{yb:.1f}"/>')

    # 4) GPU cells: rank (top) + d{shard},{pp-stage} (bottom), colored by data shard
    for r in range(world):
        _, xc, cy = geom(r)
        cx = xc - _SVG_CW / 2
        s = shard_of.get(r, 0)
        st = stage_of.get(r, 0)
        fill, stroke = _svg_color(s)
        body.append(
            f'<rect x="{cx:.1f}" y="{cy:.1f}" width="{_SVG_CW}" height="{_SVG_CH}" '
            f'rx="5" fill="{fill}" stroke="{stroke}" stroke-width="1"/>'
            f'<text class="rank" x="{xc:.1f}" y="{cy + 15:.1f}">{r}</text>'
            f'<text class="gnum" x="{xc:.1f}" y="{cy + 28:.1f}" fill="{stroke}">'
            f'd{s},{st}</text>'
        )

    # 5) DP all-reduce links, drawn ON TOP but ROUTED THROUGH THE GAPS so they
    #    never cross a cell's numbers: same-row hops bulge down into the node gap;
    #    row-to-row hops S-curve from one node's bottom edge to the next node's top.
    for g in ar_groups:
        pts = [geom(r) for r in g]
        for (ra, xa, ya), (rb, xb, yb) in zip(pts, pts[1:]):
            if ra == rb:                        # same node row: shallow arc in the gap
                yl = ya + _SVG_CH
                bulge = min(_SVG_NODE_GAP * 0.55, 13)
                body.append(
                    f'<path class="flow" d="M{xa:.1f},{yl:.1f} '
                    f'Q{(xa + xb) / 2:.1f},{yl + bulge:.1f} {xb:.1f},{yl:.1f}"/>'
                )
            else:                               # cross-node: bottom edge -> top edge
                y1, y2 = ya + _SVG_CH, yb
                ym = (y1 + y2) / 2
                body.append(
                    f'<path class="flow" d="M{xa:.1f},{y1:.1f} '
                    f'C{xa:.1f},{ym:.1f} {xb:.1f},{ym:.1f} {xb:.1f},{y2:.1f}"/>'
                )

    height = grid_top + nodes * node_pitch - _SVG_NODE_GAP + _SVG_PAD
    return "".join(body), width, height


_SCHED_CW, _SCHED_RH, _SCHED_CH = 16, 32, 20  # cell width / row pitch / box height

# Per-phase background tint (behind the F/B cells) so warmup / steady / cooldown
# are visually separable; the fill carries the phase, the cell carries F vs B.
_PHASE_BAND = {
    "warmup":   "hsl(45 90% 86%)",   # amber
    "steady":   "hsl(145 42% 88%)",  # green
    "cooldown": "hsl(280 48% 90%)",  # purple
}
_PHASE_LABEL = {"warmup": "warmup (fill)", "steady": "steady 1F1B",
                "cooldown": "cooldown (drain)"}


def _sched_color(kind: str, chunk: int) -> tuple[str, str]:
    """Fill + stroke for a schedule cell: Forward = blue, Backward = orange; the
    vpp chunk shifts lightness (chunk 0 lighter, chunk 1+ darker)."""
    h, s = (212, 60) if kind == "F" else (28, 80)
    lig = 68 - chunk * 15
    return f"hsl({h} {s}% {lig}%)", f"hsl({h} {s}% {max(24, lig - 30)}%)"


def _schedule_svg_content(cfg: Config, m: int, y0: float) -> tuple[str, int, float]:
    """Gantt of the 1F1B pipeline schedule for one pp group, drawn from y0 down.
    Returns (svg_body, width, bottom_y)."""
    p = cfg.pp
    v = cfg.layout.vpp if cfg.layout else 1
    ops, mk = pp_schedule(p, v, m)
    cw, rh, ch = _SCHED_CW, _SCHED_RH, _SCHED_CH  # cell width / row pitch / box height
    x0 = _SVG_PAD + 96
    top = y0 + 46
    busy = 2 * p * v * m
    bubble = (p * mk - busy) / (p * mk) if mk else 0

    b: List[str] = []
    b.append(f'<text class="title" x="{_SVG_PAD}" y="{y0 + 24:.0f}">'
             f'PP 1F1B schedule — one pipeline group  (pp={p}, vpp={v}, {m} microbatches)</text>')
    b.append(f'<text class="cfg" x="{_SVG_PAD}" y="{y0 + 40:.0f}">'
             f'makespan {mk} slots · bubble {bubble * 100:.1f}% of the timeline · '
             f'F=forward B=backward'
             + (' · darker = later vpp chunk' if v > 1 else '')
             + ' · dependency-scheduled 1F1B</text>')

    # per-stage, per-phase background band (contiguous in time as ops run in order)
    span: dict = {}
    for o in ops:
        k = (o["stage"], o["phase"])
        lo, hi = span.get(k, (o["t"], o["t"] + 1))
        span[k] = (min(lo, o["t"]), max(hi, o["t"] + 1))
    for (s, ph), (lo, hi) in span.items():
        ry = top + s * rh
        b.append(f'<rect x="{x0 + lo * cw:.0f}" y="{ry:.0f}" '
                 f'width="{(hi - lo) * cw:.0f}" height="{ch}" '
                 f'fill="{_PHASE_BAND[ph]}"/>')
    # each stage is its own boxed row (bold frame), with the label at the left
    for s in range(p):
        ry = top + s * rh
        b.append(f'<rect x="{x0 - 4}" y="{ry:.0f}" width="{mk * cw + 8}" '
                 f'height="{ch}" rx="4" fill="none" stroke="#5b6672" stroke-width="1.4"/>')
        b.append(f'<text class="nodelbl" x="{_SVG_PAD}" y="{ry + ch / 2 + 4:.0f}">stage {s}</text>')
    ph_h = ch - 5
    for o in ops:
        x = x0 + o["t"] * cw
        yy = top + o["stage"] * rh + 2.5
        fill, stroke = _sched_color(o["kind"], o["chunk"])
        b.append(f'<rect x="{x:.1f}" y="{yy:.1f}" width="{cw - 1.5:.1f}" height="{ph_h}" '
                 f'rx="2" fill="{fill}" stroke="{stroke}" stroke-width="0.8"/>')
        b.append(f'<text class="rank" x="{x + (cw - 1.5) / 2:.1f}" y="{yy + ph_h - 4:.1f}" '
                 f'font-size="8.5">{o["mb"]}</text>')

    # phase legend, below the figure
    ly = top + (p - 1) * rh + ch + 26
    lx = x0
    for ph in ("warmup", "steady", "cooldown"):
        b.append(f'<rect x="{lx}" y="{ly - 11:.0f}" width="13" height="13" rx="2" '
                 f'fill="{_PHASE_BAND[ph]}" stroke="#bbb"/>')
        b.append(f'<text class="lg" x="{lx + 18}" y="{ly:.0f}">{_PHASE_LABEL[ph]}</text>')
        lx += 18 + 8 * len(_PHASE_LABEL[ph]) + 24

    bottom = ly + 12 + _SVG_PAD
    width = x0 + mk * cw + _SVG_PAD
    return "".join(b), width, bottom


def write_svg(cfg: Config, path: str, gpn: int = 8, sched_m: int | None = None):
    content, width, height = _svg_content(cfg, gpn)
    if sched_m is not None:
        sc, w2, bottom = _schedule_svg_content(cfg, sched_m, height)
        content += sc
        width = max(width, w2)
        height = bottom
    svg = (
        GROUPS_SVG_TEMPLATE
        .replace("__WIDTH__", str(width))
        .replace("__HEIGHT__", str(round(height)))
        .replace("__CONTENT__", content)
    )
    out = path if path.endswith(".svg") else path + ".svg"
    with open(out, "w") as f:
        f.write(svg)
    print(f"\nWrote DP data-flow{' + PP schedule' if sched_m else ''} SVG -> {out}")


def parse_args():
    p = argparse.ArgumentParser(
        description="Resolve per-rank parallel process groups (TP/SP/CP/DP/PP/EP) "
        "from a parallelism config, Megatron-LM style."
    )
    p.add_argument("--world", type=int, default=None, metavar="N",
                   help="Total GPU count. If omitted, computed as tp*sp*cp*dp*pp.")
    p.add_argument("--tp", type=int, default=1, help="tensor-parallel size")
    p.add_argument("--sp", type=int, default=1,
                   help="sequence-parallel size (1 = rides TP group, Megatron default; "
                        ">1 = separate axis)")
    p.add_argument("--cp", type=int, default=1, help="context-parallel size")
    p.add_argument("--pp", type=int, default=1, help="pipeline-parallel size")
    p.add_argument("--dp", type=int, default=None,
                   help="data-parallel size (derived from --world if omitted)")
    p.add_argument("--ep", type=int, default=1, help="expert-parallel size (MoE)")
    p.add_argument("--etp", type=int, default=None,
                   help="expert tensor-parallel size (default = --tp)")
    p.add_argument("--order", default="tp-sp-cp-ep-dp-pp", metavar="A-B-...",
                   help="axis order, innermost first (default tp-sp-cp-ep-dp-pp)")
    p.add_argument("--pipeline-model-parallel-layout", default=None, metavar="DSL",
                   help="Megatron pipeline layout DSL (E/t/L/m, '|', x*n, (...)*n); "
                        "virtual pipeline inferred as stages//pp")
    p.add_argument("--pp-layout", default=None, metavar="n0,n1,...",
                   help="simpler alternative: decoder layers per pp stage (len = pp)")
    p.add_argument("--rank", type=int, default=None, metavar="R",
                   help="print the full comm peer breakdown for a single rank")
    p.add_argument("--limit", type=int, default=64, metavar="N",
                   help="max ranks in the per-rank table (default 64; --csv for all)")
    p.add_argument("--csv", default=None, metavar="OUT", help="write full per-rank CSV")
    p.add_argument("--svg", default=None, metavar="OUT",
                   help="write a node-organized DP data-flow SVG (GPUs colored by data "
                        "shard; links = DP gradient all-reduce)")
    p.add_argument("--gpus-per-node", type=int, default=8, metavar="N",
                   help="GPUs per physical node for the SVG layout (default 8)")
    p.add_argument("--gbs", type=int, default=None, metavar="N",
                   help="global batch size — append a 1F1B pipeline-schedule Gantt "
                        "for one pp group to the --svg. #microbatches = gbs/(dp*mbs)")
    p.add_argument("--mbs", type=int, default=1, metavar="N",
                   help="micro-batch size for the schedule (default 1)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config(args)
    maps = {label: invert(gen.get_ranks(token)) for label, gen, token, _ in comm_kinds(cfg)}
    emb = embedding_ranks(cfg)

    print_summary(cfg)

    if args.rank is not None:
        if not (0 <= args.rank < cfg.world):
            sys.exit(f"--rank {args.rank} out of range 0..{cfg.world - 1}")
        print_rank_detail(cfg, args.rank, maps, emb)
    else:
        print_group_tables(cfg)
        print_per_rank(cfg, maps, emb, args.limit)

    if args.csv:
        write_csv(cfg, maps, emb, args.csv)

    sched_m = None
    if args.gbs is not None:
        denom = cfg.dp * args.mbs
        if args.gbs % denom != 0:
            print(f"\nWarning: gbs {args.gbs} not divisible by dp*mbs = "
                  f"{cfg.dp}*{args.mbs} = {denom}; flooring.", file=sys.stderr)
        sched_m = max(1, args.gbs // denom)
        print(f"\nPP schedule: #microbatches = gbs/(dp*mbs) = {args.gbs}/"
              f"({cfg.dp}*{args.mbs}) = {sched_m}  (pp={cfg.pp}, vpp="
              f"{cfg.layout.vpp if cfg.layout else 1})")

    if args.svg:
        write_svg(cfg, args.svg, args.gpus_per_node, sched_m)
    elif sched_m is not None:
        print("  (pass --svg OUT to render the schedule Gantt)")
