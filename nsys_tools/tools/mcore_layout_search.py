"""Search Megatron-Core pipeline layouts with CP-SAT and the timeline simulator.

The optimizer deliberately separates two concerns:

* OR-Tools CP-SAT searches the integer layer-count space using an optimistic
  compute/communication relaxation and an aggregate 1F1B scheduling proxy.
* :func:`mcore_pp_timeline.schedule_overlap` is the source of truth.  Every
  candidate and every local-search neighbour is ranked by that simulator.

A layout is represented by ``S = pp * vpp`` positive integers whose sum is the
number of transformer layers.  Transformer layers remain contiguous; ``E`` is
attached to the first chunk and ``L`` to the last chunk.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

from .mcore_pp_timeline import (
    DEFAULTS,
    load_config,
    parse_dense_layers,
    parse_recompute,
    schedule_overlap,
)
from .parallel_groups import PPLayout, _pp_program


def _ortools():
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:  # pragma: no cover - exercised on remote installs
        raise SystemExit(
            "mcore_layout_search requires Google OR-Tools: pip install ortools"
        ) from exc
    return cp_model


def parse_int_values(spec: str) -> list[int]:
    """Parse ``1,2,4-6`` into a sorted list of positive integers."""
    values: set[int] = set()
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        if "-" in token:
            lo_s, hi_s = token.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                lo, hi = hi, lo
            values.update(range(lo, hi + 1))
        else:
            values.add(int(token))
    if not values or min(values) <= 0:
        raise argparse.ArgumentTypeError("expected positive integers, e.g. 4,8,16")
    return sorted(values)


def counts_to_dsl(counts: tuple[int, ...]) -> str:
    if not counts or min(counts) < 1:
        raise ValueError("every virtual pipeline chunk must contain at least one layer")
    chunks = ["t" * n for n in counts]
    chunks[0] = "E" + chunks[0]
    chunks[-1] += "L"
    return "|".join(chunks)


def balanced_counts(layers: int, stages: int) -> tuple[int, ...]:
    q, r = divmod(layers, stages)
    if q < 1:
        raise ValueError(f"{layers} layers cannot fill {stages} non-empty chunks")
    return tuple(q + (i < r) for i in range(stages))


def _phase_model(times: dict[str, float]):
    """Return per-layer two-stream coefficients and endpoint fixed costs."""
    moe = {
        "fc": times["F"] + times["E"],
        "fm": times["D"] + times["C"],
        "bc": times["E^D"] + times["E^W"] + times["F^D"] + times["F^W"],
        "bm": times["C^"] + times["D^"],
    }
    dense = {
        "fc": times["F"] + times["M"],
        "fm": 0.0,
        "bc": times["M^D"] + times["M^W"] + times["F^D"] + times["F^W"],
        "bm": 0.0,
    }
    first = {"fc": times["V"], "fm": 0.0, "bc": times["V^"], "bm": 0.0}
    last = {"fc": times["L"], "fm": 0.0, "bc": times["L^"], "bm": 0.0}
    return moe, dense, first, last


def _scaled(value: float, scale: int) -> int:
    return int(round(value * scale))


@dataclass(frozen=True)
class SearchConfig:
    pp: int
    vpp: int
    layers: int
    microbatches: int
    dense_layers: tuple[int, ...]
    recompute: str
    times: dict[str, float]
    mb_group: int | None
    ep_overlap: bool
    defer_wgrad: bool


def _simulate_one(payload):
    cfg, counts = payload
    layout = PPLayout(cfg.pp, dsl=counts_to_dsl(counts))
    dense = set(cfg.dense_layers)
    rc_map = parse_recompute(cfg.recompute, cfg.layers)
    _, makespan, _, _ = schedule_overlap(
        layout,
        dense,
        cfg.times,
        cfg.microbatches,
        rc_map=rc_map,
        mb_group=cfg.mb_group,
        overlap=cfg.ep_overlap,
        defer_wgrad=cfg.defer_wgrad,
    )
    return counts, makespan


class SimEvaluator:
    def __init__(self, cfg: SearchConfig, workers: int):
        self.cfg = cfg
        self.workers = max(1, workers)
        self.cache: dict[tuple[int, ...], float] = {}
        self.executor = None

    def __enter__(self):
        if self.workers > 1:
            self.executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.workers
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)

    def evaluate_many(self, candidates) -> dict[tuple[int, ...], float]:
        pending = []
        seen = set()
        for value in candidates:
            counts = tuple(int(x) for x in value)
            if counts not in self.cache and counts not in seen:
                pending.append(counts)
                seen.add(counts)
        payloads = [(self.cfg, counts) for counts in pending]
        if self.executor is None:
            results = map(_simulate_one, payloads)
        else:
            results = self.executor.map(_simulate_one, payloads, chunksize=1)
        for counts, makespan in results:
            self.cache[counts] = makespan
        return {tuple(c): self.cache[tuple(c)] for c in candidates}

    def evaluate(self, counts) -> float:
        key = tuple(counts)
        self.evaluate_many([key])
        return self.cache[key]


def _add_stage_duration_vars(model, assignment, cfg, scale):
    moe, dense, first, last = _phase_model(cfg.times)
    stages = cfg.pp * cfg.vpp
    horizon = _scaled(
        cfg.microbatches
        * cfg.vpp
        * cfg.pp
        * cfg.layers
        * sum(cfg.times.values()),
        scale,
    )
    horizon = max(horizon, 1)
    duration = {kind: [] for kind in ("fc", "fm", "bc", "bm")}
    aggregate = {kind: [] for kind in ("F", "B")}
    dense_set = set(cfg.dense_layers)
    for stage in range(stages):
        for kind in duration:
            base = (first[kind] if stage == 0 else 0.0) + (
                last[kind] if stage == stages - 1 else 0.0
            )
            var = model.NewIntVar(0, horizon, f"{kind}_{stage}")
            layer_work = sum(
                _scaled((dense if layer in dense_set else moe)[kind], scale)
                * assignment[layer][stage]
                for layer in range(cfg.layers)
            )
            model.Add(var == layer_work + _scaled(base, scale))
            duration[kind].append(var)
        for op_kind, comp_key, comm_key in (
            ("F", "fc", "fm"),
            ("B", "bc", "bm"),
        ):
            var = model.NewIntVar(0, horizon, f"dur_{op_kind}_{stage}")
            model.AddMaxEquality(
                var, [duration[comp_key][stage], duration[comm_key][stage]]
            )
            aggregate[op_kind].append(var)
    return duration, aggregate, horizon


def _base_cp_model(cfg: SearchConfig, scale: int):
    cp_model = _ortools()
    model = cp_model.CpModel()
    stages = cfg.pp * cfg.vpp
    assignment = [
        [model.NewBoolVar(f"x_{layer}_{stage}") for stage in range(stages)]
        for layer in range(cfg.layers)
    ]
    for layer in range(cfg.layers):
        model.AddExactlyOne(assignment[layer])
    for layer in range(cfg.layers - 1):
        model.Add(
            sum(stage * assignment[layer][stage] for stage in range(stages))
            <= sum(stage * assignment[layer + 1][stage] for stage in range(stages))
        )
    nvars = []
    for stage in range(stages):
        nvar = model.NewIntVar(1, cfg.layers - stages + 1, f"n_{stage}")
        model.Add(
            nvar == sum(assignment[layer][stage] for layer in range(cfg.layers))
        )
        nvars.append(nvar)
    duration, aggregate, horizon = _add_stage_duration_vars(
        model, assignment, cfg, scale
    )

    resource_lb = model.NewIntVar(0, horizon, "resource_lb")
    for rank in range(cfg.pp):
        comp = cfg.microbatches * sum(
            duration["fc"][s] + duration["bc"][s]
            for s in range(rank, stages, cfg.pp)
        )
        comm = cfg.microbatches * sum(
            duration["fm"][s] + duration["bm"][s]
            for s in range(rank, stages, cfg.pp)
        )
        model.Add(resource_lb >= comp)
        model.Add(resource_lb >= comm)

    path_lb = model.NewIntVar(0, horizon, "path_lb")
    model.Add(
        path_lb
        == sum(aggregate["F"]) + sum(reversed(aggregate["B"]))
    )
    lower_bound = model.NewIntVar(0, horizon, "lower_bound")
    model.AddMaxEquality(lower_bound, [resource_lb, path_lb])
    return model, nvars, duration, aggregate, lower_bound, horizon


def solve_relaxed_lower_bound(cfg, scale, seconds, workers, seed):
    cp_model = _ortools()
    model, nvars, _, _, lower_bound, _ = _base_cp_model(cfg, scale)
    model.Minimize(lower_bound)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = seed
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT lower-bound model failed with status {status}")
    counts = tuple(solver.Value(v) for v in nvars)
    proven = status == cp_model.OPTIMAL
    bound = solver.ObjectiveValue() if proven else solver.BestObjectiveBound()
    return bound / scale, counts, proven


def _proxy_model(cfg, scale):
    model, nvars, _, aggregate, lower_bound, horizon = _base_cp_model(cfg, scale)
    ops = {}
    programs = []
    for rank in range(cfg.pp):
        prog = _pp_program(
            cfg.pp,
            cfg.vpp,
            cfg.microbatches,
            rank,
            cfg.mb_group,
            cfg.ep_overlap,
        )
        programs.append(prog)
        for kind, chunk, mb, _ in prog:
            stage = chunk * cfg.pp + rank
            key = (kind, mb, stage)
            if key in ops:
                continue
            start = model.NewIntVar(0, horizon, f"start_{kind}_{mb}_{stage}")
            end = model.NewIntVar(0, horizon, f"end_{kind}_{mb}_{stage}")
            model.Add(end == start + aggregate[kind][stage])
            ops[key] = (start, end)

    for rank, prog in enumerate(programs):
        prev = None
        for kind, chunk, mb, _ in prog:
            stage = chunk * cfg.pp + rank
            key = (kind, mb, stage)
            if prev is not None:
                model.Add(ops[key][0] >= ops[prev][1])
            prev = key

    stages = cfg.pp * cfg.vpp
    for (kind, mb, stage), (start, _) in ops.items():
        if kind == "F" and stage > 0:
            model.Add(start >= ops[("F", mb, stage - 1)][1])
        if kind == "B":
            model.Add(start >= ops[("F", mb, stage)][1])
            if stage < stages - 1:
                model.Add(start >= ops[("B", mb, stage + 1)][1])

    makespan = model.NewIntVar(0, horizon, "proxy_makespan")
    model.AddMaxEquality(makespan, [end for _, end in ops.values()])
    model.Add(makespan >= lower_bound)
    return model, nvars, makespan


def solve_proxy_candidates(cfg, scale, limit, gap, seconds, workers, seed):
    cp_model = _ortools()
    model, nvars, makespan = _proxy_model(cfg, scale)
    model.Minimize(makespan)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = seed
    status = solver.Solve(model)
    if status == cp_model.UNKNOWN:
        stages = cfg.pp * cfg.vpp
        return [balanced_counts(cfg.layers, stages)], math.nan
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT proxy model failed with status {status}")
    best_proxy = int(round(solver.ObjectiveValue()))
    best = tuple(solver.Value(v) for v in nvars)
    candidates = [best]
    deadline = time.monotonic() + seconds
    allowed = max(best_proxy, int(math.ceil(best_proxy * (1.0 + gap))))
    model.Add(makespan <= allowed)
    model.ClearObjective()
    model.AddForbiddenAssignments(nvars, [list(best)])

    while len(candidates) < limit and time.monotonic() < deadline:
        remaining = max(0.05, deadline - time.monotonic())
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = remaining
        solver.parameters.num_search_workers = workers
        solver.parameters.random_seed = seed + len(candidates)
        solver.parameters.randomize_search = True
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            break
        counts = tuple(solver.Value(v) for v in nvars)
        candidates.append(counts)
        model.AddForbiddenAssignments(nvars, [list(counts)])
    return candidates, best_proxy / scale


def local_search(evaluator, seeds, rounds):
    ranked = evaluator.evaluate_many(seeds)
    best = min(ranked, key=ranked.get)
    best_ms = ranked[best]
    for _ in range(rounds):
        neighbours = []
        for src in range(len(best)):
            if best[src] <= 1:
                continue
            for dst in range(len(best)):
                if src == dst:
                    continue
                value = list(best)
                value[src] -= 1
                value[dst] += 1
                neighbours.append(tuple(value))
        scores = evaluator.evaluate_many(neighbours)
        nxt = min(scores, key=scores.get)
        if scores[nxt] >= best_ms - 1e-9:
            break
        best, best_ms = nxt, scores[nxt]
    return best, best_ms


def write_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args():
    ap = argparse.ArgumentParser(
        description="Search MCore PP layouts with OR-Tools CP-SAT and exact SIM scoring"
    )
    ap.add_argument("--pp", required=True, help='PP values, e.g. "4" or "4,8,16"')
    ap.add_argument("--vpp", required=True, help='VPP values, e.g. "2" or "1-10"')
    ap.add_argument("--layers", type=int, default=41)
    ap.add_argument("--microbatches", type=int, default=32)
    ap.add_argument("--dense-layers", default="0")
    ap.add_argument("--recompute", default="none")
    ap.add_argument("--config", default=None, help="phase-time JSON override")
    ap.add_argument("--mb-group", type=int, default=None)
    ap.add_argument("--ep-overlap", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--defer-wgrad", action="store_true")
    ap.add_argument("--candidate-limit", type=int, default=256)
    ap.add_argument("--proxy-gap", type=float, default=0.05)
    ap.add_argument("--solver-seconds", type=float, default=120.0)
    ap.add_argument("--cp-workers", type=int, default=min(32, os.cpu_count() or 1))
    ap.add_argument("--sim-workers", type=int, default=min(64, os.cpu_count() or 1))
    ap.add_argument("--sim-top", type=int, default=16)
    ap.add_argument("--local-rounds", type=int, default=20)
    ap.add_argument("--time-scale", type=int, default=10_000, help="integer ticks/ms")
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--json", default=None, metavar="OUT")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pp_values = parse_int_values(args.pp)
    vpp_values = parse_int_values(args.vpp)
    if args.layers < 1 or args.microbatches < 1:
        raise SystemExit("--layers and --microbatches must be positive")
    if args.candidate_limit < 1 or args.sim_top < 1:
        raise SystemExit("--candidate-limit and --sim-top must be positive")
    if args.proxy_gap < 0:
        raise SystemExit("--proxy-gap cannot be negative")

    times = dict(DEFAULTS)
    times.update(load_config(args.config))
    dense = tuple(sorted(parse_dense_layers(args.dense_layers, args.layers)))
    if (args.recompute or "none").lower() not in ("", "none"):
        raise SystemExit(
            "the CP-SAT relaxation does not yet support recompute; use --recompute none"
        )

    results = []
    for pp in pp_values:
        for vpp in vpp_values:
            stages = pp * vpp
            if stages > args.layers:
                print(f"skip pp={pp} vpp={vpp}: {stages} chunks > {args.layers} layers")
                continue
            cfg = SearchConfig(
                pp=pp,
                vpp=vpp,
                layers=args.layers,
                microbatches=args.microbatches,
                dense_layers=dense,
                recompute=args.recompute,
                times=times,
                mb_group=args.mb_group,
                ep_overlap=args.ep_overlap,
                defer_wgrad=args.defer_wgrad,
            )
            started = time.monotonic()
            lb_ms, lb_counts, lb_proven = solve_relaxed_lower_bound(
                cfg,
                args.time_scale,
                args.solver_seconds,
                args.cp_workers,
                args.seed,
            )
            candidates, proxy_ms = solve_proxy_candidates(
                cfg,
                args.time_scale,
                args.candidate_limit,
                args.proxy_gap,
                args.solver_seconds,
                args.cp_workers,
                args.seed,
            )
            candidates.extend([balanced_counts(args.layers, stages), lb_counts])
            candidates = list(dict.fromkeys(candidates))
            with SimEvaluator(cfg, args.sim_workers) as evaluator:
                scores = evaluator.evaluate_many(candidates)
                seeds = sorted(scores, key=scores.get)[: args.sim_top]
                counts, sim_ms = local_search(evaluator, seeds, args.local_rounds)
                evaluated = len(evaluator.cache)
            elapsed = time.monotonic() - started
            row = {
                "pp": pp,
                "vpp": vpp,
                "chunks": stages,
                "layers": args.layers,
                "microbatches": args.microbatches,
                "lower_bound_ms": lb_ms,
                "lower_bound_proven": lb_proven,
                "proxy_ms": proxy_ms,
                "sim_ms": sim_ms,
                "fixed_card_cost": pp * sim_ms,
                "requested_pp4_scaled_ms": sim_ms * 4.0 / pp,
                "fixed_card_pp4_equiv_ms": sim_ms * pp / 4.0,
                "relaxation_gap": (sim_ms - lb_ms) / sim_ms,
                "counts": list(counts),
                "layout": counts_to_dsl(counts),
                "sim_evaluated": evaluated,
                "elapsed_s": elapsed,
            }
            results.append(row)
            print(
                f"pp={pp:<2} vpp={vpp:<2} sim={sim_ms:10.3f} ms  "
                f"requested-pp4={sim_ms * 4.0 / pp:10.3f}  "
                f"fixed-card-pp4={sim_ms * pp / 4.0:10.3f}  lb={lb_ms:10.3f}  "
                f"eval={evaluated:<5} elapsed={elapsed:7.1f}s"
            )
            print(f"  counts={tuple(counts)}")
            print(f"  layout={row['layout']}")

    if not results:
        raise SystemExit("no feasible PP/VPP combinations")
    results.sort(key=lambda row: (row["requested_pp4_scaled_ms"], row["sim_ms"]))
    best = results[0]
    print("=" * 78)
    print(
        f"best requested PP4-scaled result: pp={best['pp']} vpp={best['vpp']} "
        f"sim={best['sim_ms']:.3f} ms "
        f"pp4-scaled={best['requested_pp4_scaled_ms']:.3f}"
    )
    print(best["layout"])
    if args.json:
        write_json(args.json, {"best": best, "results": results})
        print(f"wrote {args.json}")
