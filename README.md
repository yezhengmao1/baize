<p align="center">
  <img src="assets/baize-logo.svg" width="160" alt="baize logo">
</p>

# baize

NVIDIA Nsight Systems performance-analysis toolkit.

Analyzes `.nsys-rep` profiles (exported to SQLite) with **kernels as the primary entity**: NVTX context is always attached through the **kernel → NVTX mapping**, never by querying `NVTX_EVENTS` as an independent first-class table.

Packaged as the installable `baize`, providing two console commands:

| Command | Purpose |
|---|---|
| `gpu-flame` | Training step-boundary detection + per-step kernel→NVTX flame graph |
| `gpu-comm` | Cross-rank communication report: NCCL collectives/P2P + DeepEP MoE all-to-all — volume / busbw / skew |

## Install

```bash
pip install -e .          # from the repo root; provides gpu-flame / gpu-comm
```

The commands are shims in `nsys_tools/cli.py`, equivalent to `python -m nsys_tools.tools.flamegraph …`.

## Prepare data: export to SQLite

```bash
nsys export --type sqlite -o out.sqlite in.nsys-rep
```

## Tool usage

### gpu-flame — step detection + flame graph

Detects training-step boundaries from the kernel→NVTX mapping (`--step-nvtx`, default `Optimizer.step`), skips warmup (`--skip-steps`, default 1), and prints the per-step window table. With `--flamegraph OUT` it builds an interactive HTML flame graph `<OUT>.html` over the post-warmup window (d3-flame-graph: click-to-zoom / search / tooltip), with two views sharing one hierarchy:

- **Sum**: width = Σ kernel GPU time; color saturation = overlap (pale = mostly hidden behind other GPU work, vivid = exposed).
- **Solo**: width = per-kernel solo time (`active_count == 1`, the part that actually blocks the GPU clock).

`--stack-depth N` truncates the CPU NVTX path depth.

```bash
gpu-flame profile.sqlite
gpu-flame profile.sqlite --flamegraph /tmp/flame
gpu-flame profile.sqlite --step-nvtx Optimizer.step --skip-steps 1 --stack-depth 6
```

Args: `gpu-flame <db.sqlite> [--step-nvtx SUBSTR] [--skip-steps N] [--stack-depth N] [--flamegraph OUT]`

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

## Dependencies

- `nsys` (NVIDIA Nsight Systems 2026.2+)
- Python 3.12+; runtime deps `numpy`, `matplotlib` (see `pyproject.toml`); `sqlite3` is stdlib
