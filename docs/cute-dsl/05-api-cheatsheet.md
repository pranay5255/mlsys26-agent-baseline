# API Cheatsheet

## Imports

```python
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
```

Stream helpers are under `cutlass.cuda`, for example:

```python
stream = cutlass.cuda.default_stream()
```

## Decorators And Compilation

- `@cute.jit`
- `@cute.kernel`
- `cute.compile(fn, *args, options=...)`
- typed compile options such as `cute.OptLevel`, `cute.KeepPTX`,
  `cute.KeepCUBIN`, `cute.GenerateLineInfo`

## Tensor And Layout

- `cute.make_layout(shape, stride=...)`
- `cute.make_tensor(iterator_or_ptr, layout)`
- `cute.make_identity_tensor(shape)`
- `cute.local_tile(tensor, tiler, coord, proj=None)`
- `cute.size`, `cute.shape`, `cute.stride`, `cute.cosize`
- `cute.assume(value, divisibility)`

## Runtime Conversion

- `from_dlpack(tensor, assumed_align=None, use_32bit_stride=False)`
- `tensor.mark_layout_dynamic(leading_dim=None)`
- `tensor.mark_compact_shape_dynamic(mode, stride_order=None, divisibility=1)`

## Device Builtins And Sync

From `cute.arch`:

- `thread_idx`, `block_idx`, `block_dim`, `grid_dim`
- `lane_idx`, `warp_idx`
- `sync_threads`, `sync_warp`
- `barrier`, `barrier_arrive`
- `cluster_idx`, `cluster_dim`, `cluster_arrive`, `cluster_wait`
- mbarrier helpers for advanced shared-memory pipelines

## Copy And MMA

From `cute.nvgpu`:

- `CopyUniversalOp`
- `MmaUniversalOp`
- TMA helper constructors for tiled gmem to smem copies

From `cute.nvgpu.warp`:

- `MmaF16BF16Op`
- ldmatrix operations for warp-level tiled loads

Generated code should only use advanced atoms when it can construct the matching
layouts and tiled operations correctly. For non-GEMM elementwise/fusion kernels,
simple CuTe tensor indexing and explicit launch/thread mapping may be safer.
