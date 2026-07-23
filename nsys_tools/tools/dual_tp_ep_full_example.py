"""Render a complete illustrative DualPipe + TP + DeepEP schedule.

This is deliberately an example rather than a performance predictor: the macro
action order and cross-stage readiness come from the real DualPipe program, while
the operator packing inside each F/B/FB/W action uses normalized example times.
"""

from html import escape
from pathlib import Path
import argparse

from .dual_pp_time import dualpipe_program, schedule


PALETTE = {
    "attn": ("#bfdbfe", "#3b82f6"),
    "expert": ("#fed7aa", "#f97316"),
    "dgrad": ("#ddd6fe", "#8b5cf6"),
    "wgrad": ("#bbf7d0", "#22c55e"),
    "tp": ("#e9d5ff", "#7c3aed"),
    "dispatch": ("#fecdd3", "#e11d48"),
    "combine": ("#fda4af", "#be123c"),
    "pp": ("#dbe4ee", "#64748b"),
}

LANES = ("compute", "TP comm", "DeepEP", "PP comm")


def _text(buf, x, y, value, size=10, fill="#334155", anchor="start", weight=400):
    buf.append(
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
        f'text-anchor="{anchor}" font-weight="{weight}">{escape(value)}</text>'
    )


def _rect(buf, x, y, w, h, fill, stroke="none", sw=1, rx=3, opacity=1.0):
    buf.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0.4):.1f}" height="{h:.1f}" '
        f'rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>'
    )


def _global_mb(phase, local_mb, microbatches):
    return local_mb + phase * (microbatches // 2)


def _action_label(event, microbatches):
    if event["kind"] == "FB":
        fm = _global_mb(event["f_phase"], event["f_mb"], microbatches)
        bm = _global_mb(event["b_phase"], event["b_mb"], microbatches)
        fa = "→" if event["f_phase"] == 0 else "←"
        ba = "→" if event["b_phase"] == 0 else "←"
        return f"FB F{fa}{fm} + B{ba}{bm}"
    phase = event["phase"]
    mb = _global_mb(phase, event["mb"], microbatches)
    arrow = "→" if phase == 0 else "←"
    return f'{event["kind"]}{arrow}{mb}'


def _plan(kind):
    """Dependency-schedule one macro action on compute, TP and EP streams."""
    nodes = {}
    queues = {"compute": [], "tp": [], "ep": []}

    def add(name, stream, duration, label, color, preds=()):
        nodes[name] = {
            "name": name,
            "stream": stream,
            "duration": duration,
            "label": label,
            "color": color,
            "preds": list(preds),
        }
        queues[stream].append(name)

    def forward():
        add("f_ag", "tp", 1.2, "F·AG", "tp")
        add("f_attn", "compute", 2.0, "F·Attn", "attn", ("f_ag",))
        add("f_rs", "tp", 1.0, "F·RS", "tp", ("f_attn",))
        add("f_dispatch", "ep", 1.6, "F·Dispatch", "dispatch", ("f_rs",))
        add("f_expert", "compute", 2.4, "F·Expert", "expert", ("f_dispatch",))
        add("f_combine", "ep", 1.6, "F·Combine", "combine", ("f_expert",))

    def backward():
        add("b_combine", "ep", 1.6, "B·Combineᵇ", "combine")
        add("b_expert_d", "compute", 2.4, "B·ExpertD", "dgrad", ("b_combine",))
        add("b_dispatch", "ep", 1.6, "B·Dispatchᵇ", "dispatch", ("b_expert_d",))
        add("b_ag", "tp", 1.2, "B·AG", "tp", ("b_dispatch",))
        add("b_attn_d", "compute", 2.6, "B·AttnD", "dgrad", ("b_ag",))
        add("b_rs", "tp", 1.0, "B·RS", "tp", ("b_attn_d",))

    if kind == "F":
        forward()
    elif kind in ("Bcore", "Bfull"):
        backward()
        if kind == "Bfull":
            add("b_w", "compute", 1.8, "B·W", "wgrad", ("b_rs",))
    elif kind == "FB":
        add("f_ag", "tp", 1.2, "F·AG", "tp")
        add("f_attn", "compute", 2.0, "F·Attn", "attn", ("f_ag",))
        add("b_combine", "ep", 1.6, "B·Combineᵇ", "combine")
        add("b_expert_d", "compute", 2.4, "B·ExpertD", "dgrad", ("b_combine",))
        add("b_expert_w", "compute", 1.0, "B·ExpertW", "wgrad", ("b_expert_d",))
        add("f_rs", "tp", 1.0, "F·RS", "tp", ("f_attn",))
        add("f_dispatch", "ep", 1.6, "F·Dispatch", "dispatch", ("f_rs",))
        add("f_expert", "compute", 2.4, "F·Expert", "expert", ("f_dispatch",))
        add("b_dispatch", "ep", 1.6, "B·Dispatchᵇ", "dispatch", ("b_expert_d",))
        add("b_ag", "tp", 1.2, "B·AG", "tp", ("b_dispatch",))
        add("b_attn_d", "compute", 2.6, "B·AttnD", "dgrad", ("b_ag",))
        add("b_attn_w", "compute", 0.8, "B·AttnW", "wgrad", ("b_attn_d",))
        add("b_rs", "tp", 1.0, "B·RS", "tp", ("b_attn_d",))
        add("f_combine", "ep", 1.6, "F·Combine", "combine", ("f_expert",))
        queues["compute"] = [
            "f_attn",
            "b_expert_d",
            "b_expert_w",
            "f_expert",
            "b_attn_d",
            "b_attn_w",
        ]
        queues["tp"] = ["f_ag", "f_rs", "b_ag", "b_rs"]
        queues["ep"] = ["b_combine", "f_dispatch", "b_dispatch", "f_combine"]
    else:
        add("w", "compute", 1.8, "deferred W", "wgrad")

    end = {}
    free = {stream: 0.0 for stream in queues}
    ptr = {stream: 0 for stream in queues}
    remaining = len(nodes)
    while remaining:
        progressed = False
        for stream, queue in queues.items():
            if ptr[stream] == len(queue):
                continue
            node = nodes[queue[ptr[stream]]]
            if all(pred in end for pred in node["preds"]):
                start = max(
                    free[stream],
                    max((end[pred] for pred in node["preds"]), default=0.0),
                )
                node["start"] = start
                node["end"] = start + node["duration"]
                end[node["name"]] = node["end"]
                free[stream] = node["end"]
                ptr[stream] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            raise RuntimeError(f"{kind} action dependency deadlock")
    return max(end.values(), default=0.0), list(nodes.values())


def _plans():
    return {kind: _plan(kind) for kind in ("F", "Bcore", "Bfull", "FB", "W")}


def render(out, pp=4, microbatches=8, tp=4, px_per_unit=9.0):
    plans = _plans()
    programs = [dualpipe_program(pp, stage, microbatches) for stage in range(pp)]
    ft = {stage: {0: plans["F"][0], 1: plans["F"][0]} for stage in range(pp)}
    bt = {stage: {0: plans["Bcore"][0], 1: plans["Bcore"][0]} for stage in range(pp)}
    wt = {stage: {0: plans["W"][0], 1: plans["W"][0]} for stage in range(pp)}
    pair = {
        stage: {(0, 1): plans["FB"][0], (1, 0): plans["FB"][0]} for stage in range(pp)
    }
    events, makespan, busy = schedule(programs, ft, bt, wt, pair)

    pad, label_w = 28, 205
    x0 = pad + label_w
    lane_h, lane_gap, stage_gap = 21, 3, 26
    group_h = len(LANES) * (lane_h + lane_gap) - lane_gap
    top = 118
    plot_w = makespan * px_per_unit
    width = x0 + plot_w + pad
    grid_bottom = top + pp * group_h + (pp - 1) * stage_gap
    height = grid_bottom + 132

    def xx(t):
        return x0 + t * px_per_unit

    def yy(stage, lane):
        return top + stage * (group_h + stage_gap) + lane * (lane_h + lane_gap)

    b = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
        'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif">',
        f'<rect width="{width:.0f}" height="{height:.0f}" fill="#fafbfc"/>',
    ]
    _text(
        b,
        pad,
        30,
        "Full DualPipe + TP + DeepEP overlap timeline",
        18,
        "#1b2733",
        weight=700,
    )
    _text(
        b,
        pad,
        52,
        f"pp={pp} · TP={tp} per stage · microbatches={microbatches} · two mirrored chunks per TP group · {len(events)} complete F/B/FB/W actions",
        12,
        "#475569",
    )
    _text(
        b,
        pad,
        72,
        "Official DualPipe action order and readiness; normalized illustrative operator times (not measured performance).",
        11,
        "#64748b",
    )
    _text(
        b,
        pad,
        92,
        "warmup  →  bidirectional paired steady state (FB)  →  backward / deferred-wgrad drain",
        10.5,
        "#9a5b13",
        weight=700,
    )

    tick_step = 10
    for tick in range(0, int(makespan) + 1, tick_step):
        xpos = xx(tick)
        b.append(
            f'<line x1="{xpos:.1f}" y1="{top-5}" x2="{xpos:.1f}" y2="{grid_bottom}" '
            'stroke="#e5e9ef" stroke-width="1"/>'
        )
        _text(b, xpos, top - 10, f"{tick}u", 9.5, "#94a3b8", "middle")

    for stage in range(pp):
        for lane, name in enumerate(LANES):
            ypos = yy(stage, lane)
            tint = "#ffffff" if lane == 0 else "#f4f6f9"
            _rect(b, x0 - 3, ypos, plot_w + 6, lane_h, tint, "#cfd6de", 1, 4)
            _text(b, x0 - 9, ypos + 14.5, name, 8.5, "#84909d", "end")
        base = yy(stage, 0)
        mirror = pp - 1 - stage
        _text(b, pad, base + 12, f"pp {stage} / TP{tp}", 12.5, "#1b2733", weight=700)
        _text(b, pad, base + 29, f"m0 head stage {stage} →", 9, "#475569")
        _text(b, pad, base + 43, f"m1 tail stage {mirror} ←", 9, "#475569")
        bubble = (makespan - busy[stage]) / makespan * 100
        _text(
            b, pad, base + 61, f"macro bubble {bubble:.1f}%", 8.5, "#b45309", weight=700
        )
        _text(b, pad, base + 75, "TP: NVLink · EP: RDMA", 8, "#64748b")

    kind_tint = {"F": "#eff6ff", "B": "#f5f3ff", "FB": "#fff7ed", "W": "#f0fdf4"}
    kind_stroke = {"F": "#93c5fd", "B": "#c4b5fd", "FB": "#fb923c", "W": "#86efac"}
    for event in sorted(events, key=lambda item: (item["stage"], item["start"])):
        stage = event["stage"]
        start, duration = event["start"], event["dur"]
        xpos, event_w = xx(start), duration * px_per_unit
        # A faint macro boundary makes the complete DualPipe program readable.
        _rect(
            b,
            xpos,
            yy(stage, 0),
            event_w,
            3 * (lane_h + lane_gap) - lane_gap,
            kind_tint[event["kind"]],
            kind_stroke[event["kind"]],
            0.8,
            3,
            0.30,
        )
        action = _action_label(event, microbatches)
        _text(
            b,
            xpos + event_w / 2,
            yy(stage, 0) + 8,
            action,
            6.7,
            "#334155",
            "middle",
            700,
        )
        if event["kind"] == "B":
            plan_key = "Bcore" if event.get("defer") else "Bfull"
        else:
            plan_key = event["kind"]
        plan_duration, plan_nodes = plans[plan_key]
        if abs(plan_duration - duration) > 1e-9:
            raise RuntimeError(
                f"{plan_key} plan duration {plan_duration} != event duration {duration}"
            )
        lane_number = {"compute": 0, "tp": 1, "ep": 2}
        for node in plan_nodes:
            lane = lane_number[node["stream"]]
            bx = xpos + node["start"] * px_per_unit
            bw = node["duration"] * px_per_unit
            by = yy(stage, lane) + 9 if lane == 0 else yy(stage, lane) + 2
            bh = lane_h - 11 if lane == 0 else lane_h - 4
            fill, stroke = PALETTE[node["color"]]
            _rect(b, bx, by, bw, bh, fill, stroke, 0.7, 2)
            if bw >= 24:
                _text(
                    b,
                    bx + bw / 2,
                    by + bh / 2 + 3,
                    node["label"],
                    6.3,
                    "#273449",
                    "middle",
                    600,
                )

        # PP sends are products of the macro action. End stages simply have no
        # neighbor in that direction; drawing only real links keeps boundaries clear.
        pp_tokens = []
        if event["kind"] in ("F", "FB"):
            fphase = event["phase"] if event["kind"] == "F" else event["f_phase"]
            peer = stage + 1 if fphase == 0 else stage - 1
            if 0 <= peer < pp:
                pp_tokens.append("act→" if fphase == 0 else "act←")
        if event["kind"] in ("B", "FB"):
            bphase = event["phase"] if event["kind"] == "B" else event["b_phase"]
            peer = stage - 1 if bphase == 0 else stage + 1
            if 0 <= peer < pp:
                pp_tokens.append("grad←" if bphase == 0 else "grad→")
        if pp_tokens:
            # Parent schedule models PP transport as hidden. Mark the exact
            # handoff instant instead of inventing a send before data is ready.
            bx, bw = xpos + event_w - 1.0, 2.0
            by = yy(stage, 3) + 2
            fill, stroke = PALETTE["pp"]
            _rect(b, bx, by, bw, lane_h - 4, fill, stroke, 0.7, 2)

    legend_y = grid_bottom + 38
    _text(b, pad, legend_y - 17, "legend", 10.5, "#475569", weight=700)
    legend = [
        ("attn", "attention compute"),
        ("expert", "expert compute"),
        ("dgrad", "backward dgrad"),
        ("wgrad", "deferred wgrad"),
        ("tp", "TP AG/RS/AR"),
        ("dispatch", "DeepEP dispatch"),
        ("combine", "DeepEP combine"),
        ("pp", "PP P2P"),
    ]
    cursor = pad
    for color_kind, name in legend:
        fill, stroke = PALETTE[color_kind]
        _rect(b, cursor, legend_y - 10, 14, 14, fill, stroke, 1, 2)
        _text(b, cursor + 19, legend_y + 1, name, 8.8, "#475569")
        cursor += 132 if color_kind not in ("tp", "dispatch", "combine") else 148

    note_y = legend_y + 31
    _rect(b, pad, note_y - 16, width - 2 * pad, 49, "#eef6ff", "#bfdbfe", 1, 5)
    _text(b, pad + 12, note_y, "How to read:", 9.5, "#1d4ed8", weight=700)
    _text(
        b,
        pad + 86,
        note_y,
        "each outlined macro is one complete DualPipe action; FB boxes pair opposite-direction forward/backward microbatches.",
        9.5,
        "#334155",
    )
    _text(b, pad + 12, note_y + 18, "Overlap rule:", 9.5, "#1d4ed8", weight=700)
    _text(
        b,
        pad + 91,
        note_y + 18,
        "TP and DeepEP are hidden by ready compute from other layers/microbatches; shared NVLink/HBM/SM contention must be profiled, not assumed away.",
        9.5,
        "#334155",
    )

    b.append("</svg>")
    Path(out).write_text("".join(b), encoding="utf-8")
    return len(events), makespan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svg", required=True)
    parser.add_argument("--pp", type=int, default=4)
    parser.add_argument("--microbatches", type=int, default=8)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--px-per-unit", type=float, default=9.0)
    args = parser.parse_args()
    if args.pp < 2 or args.pp % 2:
        parser.error("--pp must be even and at least 2")
    if args.microbatches < 2 * args.pp or args.microbatches % 2:
        parser.error("--microbatches must be even and at least 2 * pp")
    count, makespan = render(
        args.svg, args.pp, args.microbatches, args.tp, args.px_per_unit
    )
    print(f"SVG: {args.svg} · actions={count} · makespan={makespan:.1f}u")


if __name__ == "__main__":
    main()
