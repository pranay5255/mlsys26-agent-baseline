# Tensors, Layouts, And DLPack

## CuTe Tensor Model

A CuTe tensor combines:

- an engine such as a pointer, iterator, or coordinate source
- a layout that maps logical coordinates to physical offsets

Common construction APIs include `cute.make_layout`, `cute.make_tensor`, and
`cute.make_identity_tensor`.

## PyTorch Interop

CuTe DSL integrates with frameworks through DLPack. For PyTorch tensors:

- Passing a PyTorch tensor directly to a JIT function triggers implicit
  conversion to a CuTe tensor with a dynamic layout.
- `cute.runtime.from_dlpack(tensor)` explicitly converts without copying and
  produces a static layout by default.
- Explicit conversion has overhead, so generated solutions should cache converted
  tensors or compiled executors when repeated calls have matching layout keys.

## Static Vs Dynamic Layouts

Static layouts compile with more shape information and can optimize better, but
they require recompilation for different shapes or strides. Dynamic layouts allow
one compiled executor to handle shape variation that satisfies the dynamic layout
constraints.

Use:

- `mark_layout_dynamic(leading_dim=None)` for broad dynamic layouts.
- `mark_compact_shape_dynamic(mode, stride_order=None, divisibility=1)` for
  compact tensors where only selected shape modes should be dynamic.
- `cute.assume(value, divisibility)` inside JIT code when attaching
  divisibility constraints to dynamic shapes.

## Leading Dimension Rules

`mark_layout_dynamic` preserves the stride-1 leading dimension. If `leading_dim`
is omitted, CuTe tries to infer it from strides. Ambiguous stride-1 dimensions
can fail, especially with shape-1 dimensions. Generated code should pass an
explicit leading dimension when the tensor layout is known.

## Prompt Guidance

For fixed-shape FlashInfer definitions, prefer static layout and cache by full
shape/stride/dtype. For variable-shape definitions, mark dynamic modes and cache
by a reduced key only when the generated code is truly layout-compatible.
