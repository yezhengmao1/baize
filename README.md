<p align="center">
  <img src="assets/baize-logo.svg" width="160" alt="baize logo">
</p>

# baize

NVIDIA Nsight Systems performance-analysis toolkit.

Analyzes `.nsys-rep` profiles (exported to SQLite) with **kernels as the primary entity**: NVTX context is always attached through the **kernel → NVTX mapping**, never by querying `NVTX_EVENTS` as an independent first-class table.

Packaged as the installable `baize`, providing six console commands:

| Command | Purpose |
|---|---|
| `gpu-flame` | Training step-boundary detection + per-step kernel→NVTX flame graph (Solo / Sum / Count / +idle; `--diff`) |
| `gpu-comm` | Cross-rank communication report: NCCL collectives/P2P + DeepEP MoE all-to-all — volume / busbw / skew |
| `gpu-shape` | Per-operator input-shape table over the post-warmup window (compute only) |
| `gpu-exporter` | Export a step as Chrome/Perfetto trace JSON (multi-rank, wall-clock aligned) |
| `gpu-groups` | Megatron parallel-group resolver (config-only, no profile): TP/SP/CP/DP/PP/EP |
| `sim-mcore-pp-sched` | Op-level 1F1B pipeline-timeline SVG with MoE EP-A2A overlap (config-only, no profile) |
| `sim-dual-pp-sched` | DeepSeek DualPipe bidirectional pipeline-timeline SVG (config-only, no profile) |

## Install

```bash
pip install -e .          # from the repo root; provides gpu-flame / gpu-comm / gpu-shape / gpu-exporter / gpu-groups / sim-mcore-pp-sched / sim-dual-pp-sched
```

The commands are shims in `nsys_tools/cli.py`, equivalent to `python -m nsys_tools.tools.flamegraph …`.

## Prepare data: export to SQLite

```bash
nsys export --type sqlite -o out.sqlite in.nsys-rep
```

## Tool usage

### gpu-flame — step detection + flame graph

Detects training-step boundaries from the kernel→NVTX mapping (`--step-nvtx`, default `Optimizer.step`), skips warmup (`--skip-steps`, default 1), and prints the per-step window table. With `--flamegraph OUT` it builds an interactive HTML flame graph `<OUT>.html` over the post-warmup window (d3-flame-graph: click-to-zoom / search / tooltip), with **four charts over one shared hierarchy**:

- **Solo**: width = per-activity solo time (`active_count == 1`, the part that actually blocks the GPU clock).
- **Sum**: width = Σ GPU-activity time; color saturation = overlap (pale = mostly hidden behind other GPU work, vivid = exposed).
- **Count**: width = number of GPU activities / kernel launches (wide here but narrow in Sum ⇒ launch-bound, a fusion/CUDA-graph target).
- **Sum + idle**: the Sum chart with GPU-idle added, attributed to the enclosing NVTX scope (which phase stalled).

Leaves are compute kernels plus GPU memcpy/memset (`--no-include-memcpy` to drop). `--stack-depth N` truncates the CPU NVTX path depth. **`--diff BASELINE_DB`** emits a differential flame graph (red = more GPU time now, blue = less).

```bash
gpu-flame profile.sqlite
gpu-flame profile.sqlite --flamegraph /tmp/flame
gpu-flame after.sqlite --diff before.sqlite --flamegraph /tmp/diff
```

Args: `gpu-flame <db.sqlite> [--step-nvtx SUBSTR] [--skip-steps N] [--stack-depth N] [--no-include-memcpy] [--no-attribute-idle] [--flamegraph OUT] [--diff BASELINE_DB]`

### gpu-comm — cross-rank communication report

Keys every comm kernel by its **issuing scope** — the model phase / call site that issued the comm (e.g. `_LinearBackward`, `SinkCorrectionWithCPFunc`; when only comm-plumbing frames enclose it, falls back to the outermost call such as a bare `c10d::allreduce_coalesced_` / `c10d::send` instead of `<none>`) — and aggregates across many ranks. Ranks load **in parallel** (`--jobs N`, multiprocessing; ranks are independent files).

The report has three sections, all keyed by scope:

1. **Collectives** (`ncclDevKernel` AllGather/ReduceScatter/AllReduce/Broadcast): aligned across ranks by **(comm-handle, op, seq)** — the `comm=0x…` handle is the real PG identity (globally consistent across the ranks sharing the group), so each collective instance's per-rank durations and true group size are recovered exactly (no traffic-signature guessing), then rolled up per (scope, op, width). Columns: `nranks` (real group width, e.g. 8/32/96), per-rank `calls`, Σ`vol/rank`, **busbw@floor** (achievable BW with busy-wait removed), **wait%** (`Σ(time−floor)/Σtime`, floor per collective instance), worst rank.
2. **P2P send/recv**: peer-based, no `comm=`/seq → no clean group, so only group-independent quantities per (scope, op): Σ`vol/rank`, `calls/rank`, per-rank GPU-time floor..max.
3. **EP all-to-all (DeepEP)** (`deep_ep::` dispatch/combine; volume from `FusedDispatch`/`FusedCombine` NVTX sizes): per (scope, phase), skew is computed **within EP groups** — `--ep N` partitions consecutive ranks (`rank // N`, default 8 = intranode) into groups, so the many independent 8-rank all-to-alls aren't flattened over all ranks.

- `--heatmap OUT`: write a per-rank × comm skew heatmap PNG (`--by-node` averages rows by node; `--sort {slowness,id}`).
- `--p2p-dtype-bytes N`: bytes/element for the marker-less P2P/DeepEP volume (default 2 = bf16; 1 = fp8, 4 = fp32).
- `--jobs N`: parallel-loading processes (default `min(8, cpu)`, `1` = serial).

```bash
gpu-comm rank0000.sqlite
gpu-comm rank*.sqlite --jobs 32 --ep 8 --heatmap /tmp/skew --by-node
gpu-comm rank*.sqlite --skip-steps 1 --sort slowness
```

Args: `gpu-comm <db.sqlite>... [--step-nvtx SUBSTR] [--skip-steps N] [--p2p-dtype-bytes N] [--ep N] [--heatmap OUT] [--by-node] [--sort {slowness,id}] [--jobs N]`

### gpu-shape — per-operator input-shape table

Over the post-warmup window (same step detector as `gpu-flame`), attributes each **compute** kernel to its innermost enclosing aten-op NVTX frame — the one carrying a `sizes = [[...]]` annotation (torch `emit_nvtx(record_shapes=True)`) — and rolls up per **(operator, input-shapes)**: `calls`, `kern`, Σ`gpu` time, `avg/call`, `% of compute`. Communication is excluded (NCCL by name, DeepEP by its enclosing comm scope), so the table is pure compute; the header reports the excluded comm / no-shape counts so coverage is explicit. `--html OUT` writes a self-contained interactive page (ranked bars + sortable table, grouping toggle: by (op,shapes) / by operator / by kernel).

```bash
gpu-shape profile.sqlite --top 30 --csv /tmp/shapes
gpu-shape profile.sqlite --html /tmp/shapes
```

Args: `gpu-shape <db.sqlite> [--step-nvtx SUBSTR] [--skip-steps N] [--sort {time,calls,op}] [--top N] [--csv OUT] [--html OUT]`

### gpu-exporter — Chrome/Perfetto trace JSON

Exports a step (or the whole post-warmup window) as **Chrome Trace Event Format** JSON — what `chrome://tracing` and the Perfetto UI ingest (nsys itself has no such export). Default split view: GPU streams carry kernels/memcpy/memset, NVTX ranges stay on the issuing CPU threads; `--project` instead nests each GPU op inside its enclosing NVTX scopes on the op's GPU stream. Multiple ranks merge into one trace, each its own pid namespace, **wall-clock aligned by each file's session-start UTC** so cross-node skew is visible. Extras: `--cuda-api` (default on), `--flows` (CPU-launch→GPU-kernel arrows, default on), `--comm-flows` (cross-rank NCCL P2P/collective arrows, default on), `--min-dur-ns` (drop short events). `-o OUT` writes `.json` / `.json.gz`.

```bash
gpu-exporter rank*.sqlite --step 3 -o /tmp/trace.json.gz   # open in ui.perfetto.dev
gpu-exporter rank*.sqlite --project -o /tmp/trace.json.gz
```

Args: `gpu-exporter <db.sqlite>... [--step-nvtx SUBSTR] [--skip-steps N] [--step N] [--project] [--stack-depth N] [--no-align] [--no-cuda-api] [--no-flows] [--no-comm-flows] [--min-dur-ns NS] -o OUT`

### gpu-groups — Megatron parallel-group resolver (config-only)

**No profile** — resolves the per-rank parallel process groups the way Megatron-LM lays them out, so for any rank you can see which ranks it talks to under each communication kind (TP / SP / CP / DP / PP / EP). Reimplements Megatron's `RankGenerator` (global rank = mixed-radix over the axes in `--order`). Sizes via `--tp/--sp/--cp/--pp/--dp/--ep/--etp/--world`; pipeline layer layout via `--pipeline-model-parallel-layout`/`--pp-layout` (vpp inferred). `--rank R` focuses one rank; `--csv OUT` (full per-rank); `--svg OUT` (node-organized DP data-flow diagram); `--gbs N [--mbs N]` appends a unit-time 1F1B pipeline-schedule Gantt.

```bash
gpu-groups --world 64 --tp 2 --pp 4 --ep 8 --rank 0 --svg /tmp/groups
```

Args: `gpu-groups --world N --tp N --pp N --ep N [--sp N] [--cp N] [--etp N] [--order A-B-…] [--pp-layout …] [--rank R] [--csv OUT] [--svg OUT] [--gbs N] [--mbs N]`

### sim-mcore-pp-sched — op-level 1F1B pipeline timeline with EP overlap (config-only)

**No profile** — draws a unit-anchored, wall-clock-scaled **1F1B pipeline timeline** SVG where every forward/backward op of every microbatch is decomposed into its GPU sub-phases (forward **F** attn / **D** dispatch / **E** moe-mlp / **C** combine / **M** dense-mlp / **V** embedding / **L** lm-head; backward **Fᴰ/Fᵂ Eᴰ/Eᵂ Mᴰ/Mᵂ Dʸ Cʸ Vʸ Lʸ**), each box scaled by its value and the legend carrying per-phase unit times. The schedule order is Megatron's interleaved 1F1B (reuses `gpu-groups`' `PPLayout` + `_pp_program`), and **each pp rank's own bubble %** is printed on its stage row.

**`--ep-overlap` (default on)** models MoE EP all-to-all overlap the way Megatron's `combined_1f1b.py` does: each stage splits into **two lanes — compute (attn/mlp) over comm (dispatch/combine A2A)**; in steady state a forward microbatch is paired with an *independent* backward one so the forward A2A hides behind the backward compute (and vice versa), list-scheduled by cross-stream data deps. Hidden A2A sits under a compute box, exposed A2A (warmup/cooldown) over a gap; the header reports the serial-vs-overlap makespan saving. `--no-ep-overlap` gives the serial single-lane view.

Per-phase times are **params/config, no CSV**: built-in defaults, overridden by a `--config` JSON (keys = phase tokens, e.g. `{"F":1.4,"E":1.59,"E^D":1.9}`) and/or `--t-<phase>` flags (precedence `--t-*` > `--config` > default). `--dump-config` prints the effective times as ready-to-edit JSON.

```bash
sim-mcore-pp-sched --pp 4 --pipeline-model-parallel-layout "Ett|tttttt|tttttt|tttttt|tttttt|tttttt|tttttt|tttL" \
          --microbatches 32 --svg /tmp/pp_timeline               # EP overlap on by default
sim-mcore-pp-sched --pp 4 --pipeline-model-parallel-layout "…" --dump-config > times.json
sim-mcore-pp-sched --pp 4 --pipeline-model-parallel-layout "…" --config times.json --svg /tmp/pp_timeline
```

Args: `sim-mcore-pp-sched --pp N --pipeline-model-parallel-layout "DSL" [--microbatches N] [--unit PHASE] [--ep-overlap|--no-ep-overlap] [--dense-layers IDS] [--config times.json] [--t-F MS …] [--dump-config] [--px-per-unit PX] --svg OUT`

## Dependencies

- `nsys` (NVIDIA Nsight Systems 2026.2+)
- Python 3.12+; runtime deps `numpy`, `matplotlib` (see `pyproject.toml`); `sqlite3` is stdlib
