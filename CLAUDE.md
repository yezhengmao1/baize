# CLAUDE.md

Working directory for the **nvidia-nsys-analyzer** agent. Python tools for analyzing NVIDIA Nsight Systems profiling data (`.nsys-rep` → SQLite).

**Status**: Installable package `baize` with three console entry points — `gpu-flame` (per-step kernel→NVTX flame graph), `gpu-comm` (cross-rank communication — NCCL collectives/P2P + DeepEP MoE all-to-all: volume / busbw / skew), and `gpu-shape` (per-operator input-shape table, compute only). (The earlier suite `perf_stat.py`, `kernel_analysis.py`, `comm_bw.py`, … and the standalone `report.py`/`kernel_inspect.py`/`comm_inspect.py` were removed.)

**Data model rule**: kernels are the primary entity; NVTX context is attached via the kernel->NVTX mapping (`utils.nvtx.NvtxIndex`, or `utils.comm` for the NCCL-kernel→op join). Tools may reverse-look-up kernels through this mapping but must not query `NVTX_EVENTS` as an independent first-class table.

## Environment

- **nsys**: `/usr/local/bin/nsys` (2026.2.1); convert with `nsys export --type sqlite -o out.sqlite in.nsys-rep`.
- **Python**: 3.12. Runtime deps: `numpy`, `matplotlib` (declared in `pyproject.toml`); `sqlite3` is stdlib.

## Install

```bash
pip install -e .          # from repo root; provides gpu-flame / gpu-comm / gpu-shape
```

Entry points run the tools' `if __name__ == "__main__":` blocks via small shims in `nsys_tools/cli.py` (so the CLI modules keep that style). Equivalent: `python -m nsys_tools.tools.flamegraph …`.

## Layout

```
pyproject.toml               # package + [project.scripts] gpu-flame / gpu-comm / gpu-shape
nsys_tools/
├── cli.py                   # entry-point shims (runpy -> tools' __main__)
├── tools/
│   ├── flamegraph.py        # gpu-flame: step detector + flame graph (incl. renderer: StackNode, write_html)
│   ├── gpu_comm.py          # gpu-comm: cross-rank NCCL volume / busbw / skew (incl. render_heatmap)
│   └── kernel_shapes.py     # gpu-shape: per-operator input-shape table (extract_sizes, op_frame, build_shape_table)
├── utils/
│   ├── common.py            # open_db, get_rank, human_ns, truncate, require_kernel_table
│   ├── kernel.py            # KERNEL_SQL
│   ├── nvtx.py              # NvtxEvent, NvtxIndex
│   └── comm.py              # CommEvent + load_comm_in_window (NCCL + DeepEP kernel→op/scope join)
└── templates/
    └── flamegraph.html      # d3-flame-graph HTML template (package data)
```

Cross-module imports are package-relative (`from .flame import …`, `from ..utils.common import …`); no `sys.path` hacks.

## Tools

| Binary | Purpose | Usage |
|---|---|---|
| `gpu-flame` | Step-boundary detector + optional flame graph. Detects step boundaries from the kernel->NVTX mapping (`--step-nvtx SUBSTR`, default `Optimizer.step`), skips warmup (`--skip-steps N`, default 1), prints the per-step window table. With `--flamegraph OUT` builds an interactive HTML flame graph (`<OUT>.html`, d3-flame-graph) with **three charts over one shared hierarchy**: **Solo** (non-overlap; widths = Σ per-activity solo time / `active_count==1`, the portion blocking the GPU clock), **Sum** (overlap; widths = Σ GPU-activity time, color saturation encodes overlap: vivid = exposed, pale = hidden), and **Sum + idle** (the Sum chart with GPU-idle added). `sum→solo` shrink = overlapped/hidden, equal = exposed; tooltip sum/solo/overlap%. `--stack-depth N` truncates the CPU NVTX path. Leaves are compute kernels **plus GPU memcpy (H2D/D2H/D2D) and memset** (`--include-memcpy`/`--no-include-memcpy`, default on) — these live in the separate `CUPTI_ACTIVITY_KIND_MEMCPY`/`_MEMSET` tables (joined to their launch API via `correlationId`, same NVTX mapping as kernels), colored green/purple; with them off, memcpy-only spans fall into idle. **GPU idle appears only in the third chart**: gaps are attributed to the NVTX scope enclosing each gap (`--attribute-idle`/`--no-attribute-idle`, default on) — magenta `<idle>` leaves so you can see *which phase* the GPU stalled in (gaps resolved via the same sweep-line as kernels, keyed on GPU wall-time → approximate under heavy CPU-ahead async); `--no-attribute-idle` makes it one root block. The Solo/Sum charts are pure GPU activity (no idle). Data model: `load_flame_tree` returns `(tree, tree_idle)` — `tree_idle` is `tree.clone()` + idle; `write_html` emits both as `__DATA__`/`__DATA_IDLE__`. **`--diff BASELINE_DB`** emits a differential flame graph instead: `db` is the "after", BASELINE the "before", values per-step-normalized, each frame colored by delta (red = more GPU time now, blue = less), two views = before/after widths so removed and added frames are both visible (renderer: `write_diff_html` + `templates/flamegraph_diff.html`). | `gpu-flame <db.sqlite> [--step-nvtx SUBSTR] [--skip-steps N] [--stack-depth N] [--no-include-memcpy] [--no-attribute-idle] [--flamegraph OUT] [--diff BASELINE_DB]` |
| `gpu-shape` | Per-operator input-shape table over the post-warmup window, **compute only** (communication excluded). Reuses `gpu-flame`'s step detector (`--step-nvtx`/`--skip-steps`) to pick the window, reads every kernel via `KERNEL_SQL`, and attributes each to its **innermost enclosing aten-op NVTX frame** — the frame carrying a `sizes = [[...]]` shape annotation (torch `emit_nvtx(record_shapes=True)`), resolved through the kernel→NVTX-stack mapping. `op_frame` skips comm-plumbing frames (`nccl`/`c10d::`/`record_param_comms`/`NcclGroup`/`deep_ep`/`Fused{Dispatch,Combine}`) and `ncclDevKernel*` kernels are dropped outright, so the table is pure compute. One op **invocation** = one op-frame instance `(op, shapes, start, end)`; instances roll up per **(operator, input-shapes)** → `calls` (invocations), `kern` (kernels), Σ`gpu` time, `avg/call`, `%` of compute GPU time. Header reports excluded comm / no-shape kernel counts+time so coverage is explicit. `--sort {time,calls,op}`, `--top N`, `--csv OUT` (full untruncated rows). | `gpu-shape <db.sqlite> [--step-nvtx SUBSTR] [--skip-steps N] [--sort {time,calls,op}] [--top N] [--csv OUT]` |
| `gpu-comm` | Cross-rank communication report over many ranks. Loads each rank's comm kernels **in parallel** (`--jobs N`, multiprocessing; ranks are independent files). Every comm row is keyed by its **issuing NVTX scope** — the model phase / call site that issued it (e.g. `_LinearBackward`, `SinkCorrectionWithCPFunc`, or a bare `c10d::allreduce_coalesced_` / `c10d::send` when no model module wraps it), resolved via the kernel→NVTX-stack mapping and **falling back to the outermost frame** when every enclosing frame is comm plumbing (`nccl`/`c10d::`/`record_param_comms`/`NcclGroup`). Three sections: **(1) Collectives** (`ncclDevKernel` AllGather/ReduceScatter/AllReduce/Broadcast): aligned across ranks by **(comm-handle, op, seq)** — the `comm=0x…` handle is the real PG identity (globally consistent across the ranks sharing the group), so each collective instance's per-rank durations and true group size are recovered exactly (no traffic-signature heuristic), then rolled up per (scope, op, width) → `nranks` (real group width, e.g. 8/32/96), per-rank `calls`, Σ`vol/rank` (per rank, ÷ group count), **busbw@floor** (`factor(op,nranks)×bytes / min-time-across-ranks`, busy-wait removed → achievable BW), **wait%** (`Σ(time−floor)/Σtime`, floor = min within each collective instance) + worst rank. **(2) P2P send/recv**: peer-based, no `comm=`/seq → no clean group, so only group-independent quantities per (scope, op): Σ`vol/rank`, `calls/rank`, per-rank GPU-time floor..max. **(3) EP all-to-all (DeepEP)** (`deep_ep::` dispatch/combine, incl. notify/layout helpers; volume from `FusedDispatch`/`FusedCombine` NVTX sizes): per (scope, phase), skew computed **within EP groups** of `--ep N` consecutive ranks (`rank//N`, default 8 = intranode) — a flat all-ranks floor would conflate the independent groups. `--p2p-dtype-bytes N` sizes the marker-less P2P/DeepEP volume (default 2 = bf16; 1 = fp8, 4 = fp32). `--heatmap OUT` writes a per-rank×comm skew PNG (`--by-node` averages rows by node; `--sort {slowness,id}`). | `gpu-comm <db.sqlite>... [--step-nvtx SUBSTR] [--skip-steps N] [--p2p-dtype-bytes N] [--ep N] [--heatmap OUT] [--by-node] [--sort {slowness,id}] [--jobs N]` |

Example:
```bash
gpu-flame profile.sqlite --flamegraph /tmp/flame
gpu-comm rank*.sqlite --jobs 32 --ep 8 --heatmap /tmp/skew --by-node
gpu-shape profile.sqlite --top 30 --csv /tmp/shapes
```

## Key SQLite Tables

| Table | Contents |
|---|---|
| `CUPTI_ACTIVITY_KIND_KERNEL` | GPU kernels (start, end, shortName, grid/block dims, regs, sharedMem, correlationId) |
| `CUPTI_ACTIVITY_KIND_RUNTIME` | CUDA API calls (nameId, start, end, correlationId) — joins kernels to launch site |
| `NVTX_EVENTS` | NVTX markers (textId or inline text, start, end, globalTid) |
| `StringIds` | String lookup (id → value) |
| `TARGET_INFO_SYSTEM_ENV` | `DeviceEnvironment` env dump — `RANK=` (global rank), `HOSTNAME=` (node) |

## Conventions

- `utils.common.open_db(path)` opens SQLite read-only with `sqlite3.Row` factory.
- `utils.kernel.KERNEL_SQL` is the canonical kernel+launch-API join; append `WHERE …` / `ORDER BY …` per query.
- `utils.nvtx.NvtxIndex`: `matches(substr)` finds frames by name (step partitioning); `iter_stacks(api_intervals)` bulk-resolves each kernel's enclosing stack via sweep-line.
- `utils.comm.load_comm_in_window(conn, idx, ws, we, dtype_bytes=2)` returns one `CommEvent(op, comm, seq, algo, bytes, peer, gpu_dur, gpu_start, scope)` per comm kernel that starts in `[ws, we]` — both `ncclDevKernel` (op/PG-handle/algo/bytes/peer from the `nccl:<op>` range + `comm=` marker) and **DeepEP** MoE all-to-all (`deep_ep::{intra,inter}node::{dispatch,combine,notify,…}`; op `Dispatch`/`Combine`, `comm=None`, `algo` = topology, volume from the enclosing `FusedDispatch`/`FusedCombine` range × `dtype_bytes`). `scope` is the issuing NVTX scope from `_scope_from_stack` (innermost non-plumbing frame, else the outermost frame). The caller builds the `NvtxIndex` + step window and passes them in (kernel-first, NVTX via the mapping). The basis for `gpu-comm`.
- CLI tools live in `tools/`, keep an `if __name__ == "__main__":` block, use package-relative imports, and get a `gpu-*` console entry point via `cli.py` + `[project.scripts]`. The flame-graph renderer (StackNode / write_html) lives in `flamegraph.py`; both `gpu_comm.py` and `kernel_shapes.py` reuse only its step-window helpers (`detect_steps` / `compute_step_windows`; `kernel_shapes.py` also reuses `normalize` + `KERNELS_IN_WINDOW_SQL`), loads ranks in parallel via `load_all` (`ProcessPoolExecutor`), and prints three scope-keyed sections (`_print_collectives` / `_print_p2p` / `_print_ep`). The heatmap renderer (`render_heatmap` in `gpu_comm.py`) is matplotlib-only and kept off the data path.
