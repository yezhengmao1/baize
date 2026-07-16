"""Wall-clock simulator for DeepSeek-V3/R1's bidirectional DualPipe schedule.

Unlike 1F1B, this ports the public ``deepseek-ai/DualPipe`` program: batches
enter from both ends of an even PP line; each rank owns two model modules; F0/F1/B0/B1
and deferred W actions follow the official eight warmup/steady/drain sections.
A paired F+B uses ideal operator overlap (max(F, B)); PP/EP transport is assumed
hidden by DualPipe's overlap hook.
"""

import argparse
import json
import sys
from pathlib import Path

from . import mcore_pp_timeline as pp
from .parallel_groups import PPLayout, parse_layout_dsl


class DualLayout:
    """Two mirrored full-model pipelines, matching the official DualPipe example."""

    def __init__(self, pp_size, dsl):
        stages = parse_layout_dsl(dsl)
        if len(stages) != pp_size:
            raise SystemExit(
                f"DualPipe --pp-layout must contain exactly pp={pp_size} stages; "
                f"got {len(stages)}. Supply one ordinary pipeline layout; "
                "the mirrored second module is created automatically."
            )
        embed_at = [i for i, stage in enumerate(stages) if "E" in stage]
        loss_at = [i for i, stage in enumerate(stages) if "L" in stage]
        if embed_at != [0] or loss_at != [pp_size - 1]:
            raise SystemExit(
                "DualPipe base layout requires E only on stage 0 and L only on "
                f"stage {pp_size - 1}; got E={embed_at}, L={loss_at}"
            )
        base = PPLayout(pp_size, dsl=dsl)
        self.pp = pp_size
        self.num_layers = base.num_layers
        # module0[d] is normal stage d; module1[d] is its mirrored model stage.
        self.by_pp = [
            [base.by_pp[d][0], base.by_pp[pp_size - 1 - d][0]] for d in range(pp_size)
        ]


def build_dual_durations(layout, dense, t, rc):
    """Phase segments keyed by module*pp+rank, without interleaved scheduling."""
    sf, sb = {}, {}
    for module in range(2):
        for stage in range(layout.pp):
            fsegs, bsegs = pp.chunk_segments(layout.by_pp[stage][module], dense, t, rc)
            sf[module * layout.pp + stage] = fsegs
            sb[module * layout.pp + stage] = bsegs
    return sf, sb


def _add(a, fc, bc, pending, kind, phase=None, defer=False):
    if kind == "F":
        mb = fc[phase]
        fc[phase] += 1
        a.append({"kind": kind, "phase": phase, "mb": mb})
    elif kind == "B":
        mb = bc[phase]
        bc[phase] += 1
        a.append({"kind": kind, "phase": phase, "mb": mb, "defer": defer})
        if defer:
            pending.append((phase, mb))
    elif kind == "FB":
        fp, bp = phase
        fm, bm = fc[fp], bc[bp]
        fc[fp] += 1
        bc[bp] += 1
        a.append({"kind": kind, "f_phase": fp, "f_mb": fm, "b_phase": bp, "b_mb": bm})
    else:
        if not pending:
            raise ValueError("W requested before deferred backward")
        phase, mb = pending.pop(0)  # official WeightGradStore.pop() FIFO
        a.append({"kind": "W", "phase": phase, "mb": mb})


def dualpipe_program(p, d, m):
    """Compute actions from the official DualPipe.step() eight sections."""
    half, hr = p // 2, min(d, p - 1 - d)
    mid, hm = d in (half - 1, half), m // 2
    a, fc, bc, pending = [], [0, 0], [0, 0], []
    for _ in range((half - hr - 1) * 2):
        _add(a, fc, bc, pending, "F", 0)
    for _ in range(hr + 1):
        _add(a, fc, bc, pending, "F", 0)
        _add(a, fc, bc, pending, "F", 1)
    for _ in range(half - hr - 1):
        _add(a, fc, bc, pending, "B", 1, True)
        _add(a, fc, bc, pending, "W")
        _add(a, fc, bc, pending, "F", 1)
    for i in range(hm - p + hr + 1):
        if i == 0 and mid:
            _add(a, fc, bc, pending, "F", 0)
            _add(a, fc, bc, pending, "B", 1)
        else:
            _add(a, fc, bc, pending, "FB", (0, 1))
        _add(a, fc, bc, pending, "FB", (1, 0))
    for _ in range(half - hr - 1):
        _add(a, fc, bc, pending, "B", 1)
        _add(a, fc, bc, pending, "FB", (1, 0))
    enabled = False
    for i in range(hr + 1):
        if i == (hr + 1) // 2 and hr % 2:
            enabled = True
        _add(a, fc, bc, pending, "B", 1, enabled)
        if i == (hr + 1) // 2 and not hr % 2:
            enabled = True
        _add(a, fc, bc, pending, "B", 0, enabled)
    for _ in range(half - hr - 1):
        _add(a, fc, bc, pending, "W")
        _add(a, fc, bc, pending, "B", 0, True)
    while pending:
        _add(a, fc, bc, pending, "W")
    # The reference code toggles its local phase on the second PP half.
    # Normalize actions to physical direction: 0 is left->right, 1 right->left.
    if d >= half:
        for action in a:
            if action["kind"] == "FB":
                action["f_phase"] ^= 1
                action["b_phase"] ^= 1
            else:
                action["phase"] ^= 1
    return a


def pair_timeline(layout, dense, t, rc, stage, f_phase, b_phase):
    """One DualPipe paired action, using mcore_pp_timeline's exact two-stream node plan."""
    g = pp._NodeGen()
    fop = pp._fwd_op(g, 0, f_phase * layout.pp + stage, layout, dense, t)
    bop = pp._bwd_op(g, 1, b_phase * layout.pp + stage, layout, dense, t, rc)
    comp, comm = [], []
    pp._issue_combined(comp, comm, fop, bop, defer_wgrad=False)
    by_id = {n["id"]: n for n in g.nodes}
    streams = [["comp", comp, 0, 0.0], ["comm", comm, 0, 0.0]]
    end, remaining = {}, len(comp) + len(comm)
    while remaining:
        progress = False
        for stream in streams:
            name, queue, ptr, free = stream
            if ptr == len(queue):
                continue
            node = by_id[queue[ptr]]
            if all(pred in end for pred in node["preds"]):
                start = max(
                    free, max((end[pred] for pred in node["preds"]), default=0.0)
                )
                node["start"], node["end"] = start, start + node["ms"]
                end[node["id"]] = node["end"]
                stream[2], stream[3] = ptr + 1, node["end"]
                remaining -= 1
                progress = True
        if not progress:
            raise RuntimeError("paired F/B stream dependency deadlock")
    boxes = [
        {
            "tok": n["tok"],
            "kind": n["kind"],
            "stream": n["stream"],
            "start": n["start"],
            "ms": n["ms"],
        }
        for n in g.nodes
    ]
    return max((n["end"] for n in g.nodes), default=0.0), boxes


def schedule(programs, ft, bt, wt, pair_dur):
    """Dependency-time the two directions; P2P costs are intentionally hidden."""
    p = len(programs)
    ptr = [0] * p
    free = [0.0] * p
    busy = [0.0] * p
    end = {}
    out = []
    while len(out) < sum(map(len, programs)):
        progress = False
        for d in range(p):
            if ptr[d] == len(programs[d]):
                continue
            a, deps, products = programs[d][ptr[d]], [], []

            def fd(ph, mb):
                src = d - 1 if ph == 0 else d + 1
                if 0 <= src < p:
                    key = ("F", ph, mb, src)
                    if key not in end:
                        return False
                    deps.append(end[key])
                products.append(("F", ph, mb, d))
                return True

            def bd(ph, mb):
                own = ("F", ph, mb, d)
                src = d + 1 if ph == 0 else d - 1
                if own not in end:
                    return False
                deps.append(end[own])
                if 0 <= src < p:
                    key = ("B", ph, mb, src)
                    if key not in end:
                        return False
                    deps.append(end[key])
                products.append(("B", ph, mb, d))
                return True

            if a["kind"] == "F":
                ready, dur = fd(a["phase"], a["mb"]), ft[d][a["phase"]]
            elif a["kind"] == "B":
                ready = bd(a["phase"], a["mb"])
                dur = bt[d][a["phase"]] + (0 if a["defer"] else wt[d][a["phase"]])
            elif a["kind"] == "W":
                key = ("B", a["phase"], a["mb"], d)
                ready, dur = key in end, wt[d][a["phase"]]
                if ready:
                    deps.append(end[key])
            else:
                ready = fd(a["f_phase"], a["f_mb"])
                ready = ready and bd(a["b_phase"], a["b_mb"])
                dur = pair_dur[d][(a["f_phase"], a["b_phase"])]
            if not ready:
                continue
            e = dict(a, stage=d, start=max(free[d], max(deps, default=0.0)), dur=dur)
            e["end"] = e["start"] + dur
            for key in products:
                end[key] = e["end"]
            free[d] = e["end"]
            busy[d] += dur
            ptr[d] += 1
            out.append(e)
            progress = True
        if not progress:
            raise RuntimeError("DualPipe dependency deadlock")
    return out, max(free), busy


def draw_phase_legend(b, leg_pos, t, u, pad, y):
    """The pp_timeline legend without the unrelated chunk-stripe wording."""
    b.append(
        f'<text x="{pad}" y="{y - 13}" font-size="11" font-weight="700" fill="#4b5563">phases (value in units)</text>'
    )
    for xoff, row, tok in leg_pos:
        x, yy = pad + xoff, y + row * 20
        fill, stroke = pp._color(tok)
        b.append(
            f'<rect x="{x:.1f}" y="{yy - 10}" width="13" height="13" rx="2" fill="{fill}" stroke="{stroke}"/>'
        )
        if pp._is_rc(tok):
            b.append(
                f'<rect x="{x:.1f}" y="{yy - 10}" width="13" height="13" rx="2" fill="url(#rc)"/>'
            )
        b.append(
            f'<text x="{x + 18:.1f}" y="{yy}" font-size="10.5" fill="#374151">{pp._esc(pp.LABEL[tok])} ({u(pp._ttime(t, tok)):.2f}u)</text>'
        )


def validate_forward_flow(events, pp_size):
    """Every non-origin F must start after the same mb arrives from its neighbor."""
    finished = {}
    for event in events:
        forwards = []
        if event["kind"] == "F":
            forwards.append((event["phase"], event["mb"]))
        elif event["kind"] == "FB":
            forwards.append((event["f_phase"], event["f_mb"]))
        for phase, mb in forwards:
            upstream = event["stage"] - 1 if phase == 0 else event["stage"] + 1
            if 0 <= upstream < pp_size:
                key = (phase, mb, upstream)
                if key not in finished or event["start"] + 1e-9 < finished[key]:
                    raise RuntimeError(
                        f"F dir{phase} mb{mb} pp{event['stage']} starts before recv from pp{upstream}"
                    )
            finished[(phase, mb, event["stage"])] = event["end"]


def render_svg(
    events,
    mk,
    unit,
    px,
    out,
    t,
    sf,
    sb,
    pair_layouts,
    microbatches,
    microbatch_size,
    layout,
):
    """Render the actual compute/communication streams with full phase boxes."""
    p = max(e["stage"] for e in events) + 1 if events else 0
    ppu = px / unit
    PAD, LBL = 26, 165
    LANE_H, LANE_GAP, ROW_GAP = 26, 3, 20
    ROW_H, row_pitch, top = LANE_H * 2 + LANE_GAP, LANE_H * 2 + LANE_GAP + ROW_GAP, 104
    x0, plot_w = PAD + LBL, mk * ppu
    width, grid_bot = x0 + plot_w + PAD, top + p * row_pitch - ROW_GAP

    def u(ms):
        return ms / unit

    extra = [
        tok
        for tok in pp.RECOMP_ORDER
        if any(tok == k for segs in sb.values() for k, _ in segs)
    ]
    leg_pos, leg_rows = pp._legend_flow(t, u, plot_w, extra)
    height = grid_bot + 34 + leg_rows * 20 + PAD

    def is_comm(tok):
        return tok in {"D", "C", "D^", "C^", "D~", "C~"}

    def serial_boxes(segs, kind):
        at, boxes = 0.0, []
        for tok, ms in segs:
            boxes.append(
                {
                    "tok": tok,
                    "kind": kind,
                    "stream": "comm" if is_comm(tok) else "comp",
                    "start": at,
                    "ms": ms,
                }
            )
            at += ms
        return boxes

    def bsegs(stage, phase, mode):
        segs = sb[phase * p + stage]
        if mode == "core":
            return [(k, v) for k, v in segs if not k.endswith("^W")]
        if mode == "w":
            return [(k, v) for k, v in segs if k.endswith("^W")]
        return segs

    def boxes_for(e):
        if e["kind"] == "FB":
            return pair_layouts[e["stage"]][(e["f_phase"], e["b_phase"])][1]
        if e["kind"] == "F":
            return serial_boxes(sf[e["phase"] * p + e["stage"]], "F")
        if e["kind"] == "B":
            return serial_boxes(
                bsegs(e["stage"], e["phase"], "core" if e["defer"] else "full"), "B"
            )
        return serial_boxes(bsegs(e["stage"], e["phase"], "w"), "B")

    comp_busy, comm_busy = [0.0] * p, [0.0] * p
    for e in events:
        for node in boxes_for(e):
            (comp_busy if node["stream"] == "comp" else comm_busy)[e["stage"]] += node[
                "ms"
            ]
    ideal = max(comp_busy) if comp_busy else 0.0
    bubble = (mk - ideal) / mk if mk else 0.0
    b = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" font-family="-apple-system,Segoe UI,Roboto,sans-serif">',
        f'<rect width="{width:.0f}" height="{height:.0f}" fill="#fafbfc"/>',
        f'<text x="{PAD}" y="32" font-size="17" font-weight="700" fill="#1b2733">DeepSeek DualPipe + EP-overlap timeline &#8212; pp={p}, modules/rank=2</text>',
        f'<text x="{PAD}" y="54" font-size="12.5" fill="#4b5563">unit 1 = {unit:.3f} ms &#183; mbs={microbatch_size} &#183; each stage has <tspan font-weight="700">comp</tspan> over <tspan font-weight="700">comm</tspan> &#183; paired opposite-direction F/B uses the same combined node plan as mcore_pp_timeline &#183; makespan {mk / unit:.1f} u ({mk:.0f} ms) &#183; bubble {bubble * 100:.1f}%</text>',
        pp._HATCH_DEFS,
    ]
    raw = mk / unit / 12 if mk else 1
    mag = 10 ** (len(str(int(raw))) - 1) if raw >= 1 else 1
    step, tick = (max(mag, round(raw / mag) * mag) if raw else 1), 0.0
    while tick <= mk / unit + 1e-6:
        x = x0 + tick * px
        b += [
            f'<line x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{grid_bot}" stroke="#e7ebef" stroke-width="1"/>',
            f'<text x="{x:.1f}" y="{top - 10}" font-size="10" fill="#9aa4b0" text-anchor="middle">{tick:.0f}u</text>',
        ]
        tick += step
    for d in range(p):
        ry = top + d * row_pitch
        for lane, ly, tint in (
            ("comp", ry, "#ffffff"),
            ("comm", ry + LANE_H + LANE_GAP, "#f4f6f9"),
        ):
            b += [
                f'<rect x="{x0 - 3}" y="{ly:.0f}" width="{plot_w + 6:.1f}" height="{LANE_H}" rx="4" fill="{tint}" stroke="#cfd6de" stroke-width="1"/>',
                f'<text x="{x0 - 8}" y="{ly + LANE_H / 2 + 3:.0f}" font-size="8" fill="#9aa4b0" text-anchor="end">{lane}</text>',
            ]

        def module_label(module):
            chunk = layout.by_pp[d][module]
            ids = chunk["layers"]
            layers = (
                f"L{ids[0]}-{ids[-1]}"
                if len(ids) > 1
                else (f"L{ids[0]}" if ids else "-")
            )
            flags = (
                ("E," if "E" in chunk["chars"] else "")
                + layers
                + (",Loss" if "L" in chunk["chars"] else "")
            )
            return flags

        rb = (mk - comp_busy[d]) / mk if mk else 0
        b += [
            f'<text x="{PAD}" y="{ry + 12:.0f}" font-size="12.5" font-weight="700" fill="#1b2733">pp {d}</text>',
            f'<text x="{PAD}" y="{ry + 24:.0f}" font-size="8.5" fill="#475569">m0 &#8594; {module_label(0)}</text>',
            f'<text x="{PAD}" y="{ry + 34:.0f}" font-size="8.5" fill="#475569">m1 &#8592; {module_label(1)}</text>',
            f'<text x="{PAD}" y="{ry + 45:.0f}" font-size="9" font-weight="700" fill="#b4530a">bubble {rb * 100:.1f}%</text>',
            f'<text x="{PAD}" y="{ry + 53:.0f}" font-size="7" fill="#8a95a1">comp {u(comp_busy[d]):.0f}u comm {u(comm_busy[d]):.0f}u</text>',
        ]

    for e in events:
        if e["kind"] == "FB":
            phase_mb = {"F": (e["f_phase"], e["f_mb"]), "B": (e["b_phase"], e["b_mb"])}
        else:
            phase_mb = {"F" if e["kind"] == "F" else "B": (e["phase"], e["mb"])}
        for node in boxes_for(e):
            phase, local_mb = phase_mb[node["kind"]]
            mb = local_mb + phase * (microbatches // 2)
            ry = top + e["stage"] * row_pitch
            ly = ry if node["stream"] == "comp" else ry + LANE_H + LANE_GAP
            x = x0 + (e["start"] + node["start"]) * ppu
            w = node["ms"] * ppu
            tok = node["tok"]
            fill, stroke = pp._color(tok)
            if e["kind"] == "W":
                source = "deferred local wgrad"
            else:
                upstream = (
                    (e["stage"] - 1 if phase == 0 else e["stage"] + 1)
                    if node["kind"] == "F"
                    else (e["stage"] + 1 if phase == 0 else e["stage"] - 1)
                )
                source = (
                    ("origin" if node["kind"] == "F" else "loss-origin")
                    if not (0 <= upstream < p)
                    else f"recv pp{upstream}"
                )
            b.append(
                f'<rect x="{x:.2f}" y="{ly + 2:.1f}" width="{max(w, 0.4):.2f}" height="{LANE_H - 4}" fill="{fill}" stroke="{stroke}" stroke-width="0.5"><title>mb {mb} {node["kind"]} dir{phase} pp{e["stage"]} ({source}) &#183; {pp._esc(tok)} = {u(node["ms"]):.2f} u ({node["ms"]:.3f} ms)</title></rect>'
            )
            if pp._is_rc(tok):
                b.append(
                    f'<rect x="{x:.2f}" y="{ly + 2:.1f}" width="{max(w, 0.4):.2f}" height="{LANE_H - 4}" fill="url(#rc)"/>'
                )
            center = x + w / 2
            if w >= 13:
                b.append(
                    f'<text x="{center:.2f}" y="{ly + 13:.1f}" font-size="9.5" fill="#0f1b28" text-anchor="middle">{pp._esc(pp.GLYPH.get(tok, tok))}</text>'
                )
                tri = "&#9650;" if node["kind"] == "F" else "&#9660;"
                b.append(
                    f'<text x="{center:.2f}" y="{ly + 21:.1f}" font-size="6" fill="#5b6673" text-anchor="middle">{tri}{mb}</text>'
                )
            elif w >= 7:
                b.append(
                    f'<text x="{center:.2f}" y="{ly + LANE_H / 2 + 3.2:.1f}" font-size="8.5" fill="#0f1b28" text-anchor="middle">{pp._esc(pp.GLYPH.get(tok, tok))}</text>'
                )
    draw_phase_legend(b, leg_pos, t, u, PAD, grid_bot + 34)
    b.append("</svg>")
    path = out if out.endswith(".svg") else out + ".svg"
    Path(path).write_text("".join(b))
    return path


def parse_args():
    ap = argparse.ArgumentParser(
        description="DeepSeek-V3 bidirectional DualPipe wall-clock simulator"
    )
    ap.add_argument("--pp", type=int, required=True)
    ap.add_argument(
        "--pipeline-model-parallel-layout",
        "--pp-layout",
        dest="layout",
        required=True,
        metavar="DSL",
        help="one ordinary pp-stage layout; DualPipe creates the mirrored second model copy",
    )
    ap.add_argument(
        "--microbatches",
        dest="m",
        type=int,
        default=32,
        help="number of pipeline chunks (default 32)",
    )
    ap.add_argument(
        "--mbs",
        "--microbatch-size",
        dest="mb_size",
        type=int,
        choices=[2],
        default=2,
        help="micro-batch size; fixed to 2",
    )
    ap.add_argument(
        "--dense-layers",
        default="0",
        metavar="IDS",
        help="comma-separated zero-based global transformer-layer ids that are "
        'dense, e.g. "0,1,4"; default "0" (only the first layer). '
        '"none" = all MoE',
    )
    ap.add_argument("--recompute", default="none", metavar="SPEC")
    ap.add_argument("--config", metavar="times.json")
    ap.add_argument("--dump-config", action="store_true")
    ap.add_argument("--unit", default="F", metavar="PHASE")
    ap.add_argument("--unit-ms", type=float, metavar="MS")
    ap.add_argument("--px-per-unit", type=float, default=13.0, metavar="PX")
    ap.add_argument("--svg", metavar="OUT")
    for tok in pp.FWD_TOKENS + pp.BWD_TOKENS:
        ap.add_argument(
            f"--t-{pp._flag(tok)}", dest="t_" + pp._flag(tok), type=float, metavar="MS"
        )
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = pp.load_config(args.config)
    t = pp.resolve_times(args, cfg)
    if args.dump_config:
        print(
            json.dumps(
                {k: round(t[k], 4) for k in pp.FWD_TOKENS + pp.BWD_TOKENS}, indent=2
            )
        )
        sys.exit(0)
    if not args.svg:
        sys.exit("--svg OUT is required")
    layout = DualLayout(args.pp, args.layout)
    if args.pp < 2 or args.pp % 2:
        sys.exit("DualPipe requires an even --pp of at least 2")
    if args.m % 2 or args.m < 2 * args.pp:
        sys.exit("DualPipe requires even --microbatches >= 2 * pp")
    unit = args.unit_ms if args.unit_ms is not None else t.get(args.unit)
    if unit is None or unit <= 0:
        sys.exit("--unit must name a positive phase, or use --unit-ms")
    dense = pp.parse_dense_layers(args.dense_layers, layout.num_layers)
    rc = pp.parse_recompute(args.recompute, layout.num_layers)
    sf, sb = build_dual_durations(layout, dense, t, rc)
    ft = {}
    bt = {}
    wt = {}
    for d in range(args.pp):
        ft[d], bt[d], wt[d] = {}, {}, {}
        for phase in range(2):
            vs, bsegs = phase * args.pp + d, sb[phase * args.pp + d]
            ft[d][phase] = sum(v for _, v in sf[vs])
            wt[d][phase] = sum(v for k, v in bsegs if k.endswith("^W"))
            bt[d][phase] = sum(v for k, v in bsegs if not k.endswith("^W"))
    pairs = {d: {} for d in range(args.pp)}
    for d in range(args.pp):
        for fp, bp in ((0, 1), (1, 0)):
            pairs[d][(fp, bp)] = pair_timeline(layout, dense, t, rc, d, fp, bp)
    pair_dur = {d: {key: value[0] for key, value in pairs[d].items()} for d in pairs}
    events, mk, _ = schedule(
        [dualpipe_program(args.pp, d, args.m) for d in range(args.pp)],
        ft,
        bt,
        wt,
        pair_dur,
    )
    validate_forward_flow(events, args.pp)
    path = render_svg(
        events,
        mk,
        unit,
        args.px_per_unit,
        args.svg,
        t,
        sf,
        sb,
        pairs,
        args.m,
        args.mb_size,
        layout,
    )
    print(
        f"DualPipe pp={args.pp} modules/rank=2 mbs={args.mb_size} microbatches={args.m} makespan={mk / unit:.2f}u ({mk:.3f}ms)"
    )
    print(f"SVG: {path}")
