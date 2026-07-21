# Performance and capacity guidance

The target scale is about 10 million final hexahedra with fewer than 1,000
input geometry objects.  Those two numbers affect different phases: geometry
count controls rail/rule preprocessing, while the 2D cross-section size and
number of z subdivisions control the final 3D arrays.

## Scaling invariants

- Rail compatibility and snap-rule construction should operate on geometry
  lines and intervals, not on final 3D elements.
- Rail construction precomputes pairwise span/Z incompatibility for the
  geometry features as Python-integer bitsets.  Candidate insertion then uses
  bit operations, coordinate-range blocker masks, and only the current
  neighbouring rails/pinned coordinates for the order proof.  The one-time
  relation build is `O(G^2)` for `G` geometry features; with the supported
  target of fewer than 1,000 geometries this is intentionally independent of
  the final 3D mesh size.
- Topology coordinate matching remains exact even for values one ULP apart.
  Scale-aware noise handling is limited to classifying one input line's axis;
  it must never broaden rail lookup, coverage, or material boundaries.
- Generated structured meshes should derive rail node ids from the one-
  dimensional x/y axes.  They must not sort or deduplicate the full element
  connectivity merely to find a rail.
- A custom mesh that supplies trusted-looking structured metadata must still
  be verified exactly, in bounded chunks, against its row-major nodes and
  connectivity before arithmetic indexing is enabled.  Metadata is a compact
  contract, not permission to skip topology validation.
- A snap event should perform binary searches on the affected rails, update
  only the selected node spans, and validate only incident quadrilaterals.
- Whole-2D-mesh orientation validation is expected once when a mesh is
  generated or assigned.  It uses bounded chunks; it must not run once per z
  event.
- Generic custom assignment scans edge topology once to prove continuous,
  collinear feature coverage and complete mandatory stations.  It must not
  retain all mesh edges after validation.
- For an explicit BOX domain, generic custom assignment also retains four
  compact edge keys per quad during a one-time manifold/boundary/area
  partition proof.  Large row-major BOX meshes should use verified structured
  metadata to avoid this `O(E2)` edge-key transient.
- Snap-plan preflight enumerates exact z boundaries and the open intervals
  between them, but materializes only rule-touched nodes/cells.  No-op rules
  do not enlarge the touched span, identical effective rule geometries are
  deduplicated across feature lifecycles, and overlapping lifecycles for the
  same target/span are unioned before state enumeration.  Its scale is
  geometry/axis driven, not proportional to final 3D hexahedra.
- 3D node planes and hexahedron connectivity are written into final storage in
  bounded flat-final-hex chunks.  The number of Python iterations therefore
  follows bounded chunks, not the number of Z planes, and no temporary shaped
  like `(z_layers, elements_2d, 8)` is created.
- Build selector-boundary representability is preflighted from fewer than
  1,000 geometry intervals, without scanning the 2D mesh.  Actual
  area/range/hole element classification necessarily scans candidate quads,
  but does so with bounded coordinate chunks.
- When both the 2D mesh and selector count are large enough, one
  `_organize()` pass builds a temporary uniform-bin index from an actual quad
  corner.  Selector bounding boxes only prune candidates; the unchanged exact
  four-corner and hole predicates remain authoritative.  Unsupported or
  unsafe selector/index shapes fall back to the bounded full scan.  The index
  is discarded before any later snap can make it stale, and selector hit
  arrays are consumed one at a time, so retained work memory is `O(E2)` rather
  than `O(selectors * E2)`.
- Material layers use two fixed `int32` component buffers.  At each overlay,
  only the preceding active footprint is inherited; a full-domain footprint
  uses a contiguous buffer copy.  This work is bounded by emitted slab cells
  (each inherited active quad produces at least one hexahedron), not by
  `all_2D_cells * geometry_events`, and it allocates no per-layer mesh-sized
  component buffer.
- Every z subdivision is proven representable in float64 before its slab
  changes mappings or allocates output.  Final-sentinel XY synchronization is
  sparse and validates only incident 3D elements.
- A generated or assigned mesh records an authoritative baseline digest.
  Before each build, nodes and connectivity are hashed in bounded row chunks,
  after managed snap state is normalized to baseline.  This is one `O(N2 +
  E2)` safety scan per build, not one scan per z subdivision, and it does not
  allocate a second mesh-sized contiguous hash input.
- A legal active snap state retains sparse expected target values alongside
  changed node ids.  Build verifies those values exactly and substitutes
  baseline coordinates only inside bounded hash chunks, so an unmanaged edit
  to an already-snapped node cannot be hidden by normalization.
- Connectivity and structural index arrays are read-only after indexing.
  Layout/identity and structural digests make replacement or forced in-place
  mutation fail closed; the flags and digests add negligible storage compared
  with the indexed arrays themselves.
- The structured BOX generator writes broadcast X/Y columns directly into the
  final node array and produces canonical `int32` connectivity in bounded
  chunks.  It does not allocate full `meshgrid` coordinate matrices or an
  intermediate platform-integer connectivity grid.
- Material organization reuses fixed-size component buffers, clears only the
  previously active indices, computes each selector once in its priority
  pass, and shares the NORMAL volume/assignment scan.  Node-to-3D mappings are
  likewise reset only at indices used by the previous extrusion.

The complexity regression tests enforce work-set sizes and allocation reuse,
not wall-clock limits.  This keeps them stable across developer machines and
CI while still detecting an accidental full-mesh scan in the snap path.

## Raw array capacity estimate

The current in-memory representation uses 32-bit connectivity and ids and
64-bit node coordinates.  For `H` final hexahedra and `N3` final 3D nodes, the
main final arrays require approximately:

| Array group | Bytes per item | At 10 million items |
| --- | ---: | ---: |
| Hex connectivity (`8 x int32`) | 32 per hex | 320 MB |
| Hex id + component (`2 x int32`) | 8 per hex | 80 MB |
| 3D xyz + node id | 28 per node | 280 MB per 10M nodes |

In formula form, final storage is approximately `40*H + 28*N3` bytes.  Do not
assume `N3 == H`: node count depends on the footprint, disconnected regions,
and how planes are reused.  For example, a square cross-section with about one
million quads extruded through ten subdivisions has roughly 10 million hexes
and 11 million 3D nodes, or about 708 MB of final raw arrays.  A thin,
fragmented, or many-component footprint can require more than 1 GB raw for the
same hex count because it shares fewer nodes.

When every slab contains exactly one hole-free, metal-free BOX area equal to
the complete BOX domain, `build()` proves the exact final node/hex counts and
reserves both final arrays once.  This common full-domain path avoids growth
copies entirely.  General selector/material cases cannot know the selected
element count without classification, so they retain incremental growth.
That growth is capped at one million unused records, adding at most about
40 MB for hex arrays or 28 MB for node arrays at a growth boundary.  A growth
operation can temporarily hold both the old and new arrays, so this spare-
capacity bound is not itself a bound on peak RSS.

For `N2` 2D nodes and `E2` 2D quads, a generated or verified-structured base
mesh, Dragger state, and reusable snap-conflict buffers are roughly
`40*N2 + 36*E2` bytes before rail node ids.  Per-event target copies remain
proportional only to touched nodes.  Rail node-id membership can add up to
about `16*N2` bytes when every x and y station is a rail.  Structured mode
keeps per-node axis ownership out of the hot path: its axis-id maps are only
about `4*(len(x_nodes)+len(y_nodes))` bytes, and incident elements are derived
by arithmetic.

A generic custom mesh additionally keeps two node-to-rail-id arrays (about
`16*N2` bytes on a 64-bit platform), a node-to-element CSR of roughly
`8*(N2+1) + 16*E2` bytes, and custom rail node-id/span arrays that can reach
about `32*N2` bytes.  Thus worst-case all-rail custom membership is about
`48*N2` bytes beyond the common base, not `16*N2`.  Generic exact coverage,
global coordinate sorting, and orientation checks also create chunked or
mesh-sized transient work arrays during assignment.

For an explicit BOX domain, the generic partition proof temporarily stores
four `uint64` edge keys per quad (about `32*E2` bytes) before global
deduplication/counting, plus sort/unique outputs.  During concatenation the
source chunks and the contiguous key array can coexist, adding another roughly
`32*E2` bytes before unique/count temporaries; measure peak RSS rather than
budgeting only the final key array.  This is a safety cost for
arbitrary topology: it detects holes, disconnected pieces, non-manifold or
duplicate elements, incomplete physical boundaries, and area overlap.  The
verified `STRUCTURED_BOX` path proves the same rectangular partition from
strict axes and canonical connectivity without those global edge keys.

`preserve_mesh2d=True` makes exactly one full working-node copy (about
`24*N2` bytes for three float64 columns) and preserves the caller's existing
coordinates and active snap-state bookkeeping.  Multiple stacks are rejected
by default; opting into `allow_independent_bodies=True` means their 3D nodes
are intentionally separate, so touching interfaces do not share storage or
FEM topology and can increase final node count substantially.

## Structured custom mesh metadata

For a very large row-major BOX grid, attach generated-style metadata to avoid
the generic custom CSR and full edge-coverage path:

```python
mesh2d.metadata = {
    "kind": "STRUCTURED_BOX",
    "grid_shape": (len(y_nodes), len(x_nodes)),
    "x_nodes": x_nodes,
    "y_nodes": y_nodes,
}
mesher.mesh_assignment(mesh2d)
```

The same mapping can be passed explicitly as
`mesh_assignment(mesh2d, structured_metadata=metadata)`.  The implementation
does not infer this mode merely from array shapes.  It checks axis values,
grid shape, every row-major node coordinate, and canonical quadrilateral
connectivity exactly in bounded chunks.  Only after that verification does it
install direct axis rail lookup and arithmetic node-to-element adjacency.
This path avoids generic coverage edge retention, global coordinate sorting,
per-node rail maps, and CSR construction.

Without structured metadata, assignment intentionally takes the generic
safety path: exact continuous edge-chain/station coverage, BOX partition
proof when a BOX domain is declared, generic rail indexing, and CSR
construction.  This is appropriate for arbitrary quadrilateral topology but
can materially increase peak RSS when the *2D* custom mesh itself is near ten
million records.  The usual production target is ten million final 3D hexes
with a much smaller reusable 2D cross-section; do not treat those two counts
as interchangeable.  Supplied metadata that fails verification rejects the
assignment; it does not silently fall back to the generic path and conceal a
metadata/topology mismatch.  POLYGON and CYLINDER footprint partition proofs
are not implemented, so their custom assignment and build paths fail closed.

These are raw-array estimates rather than an RSS guarantee.  Plan production
runs with headroom for input geometry, classification masks, NumPy
temporaries, multiple objects, array growth while old and new allocations
coexist, and the Python runtime.  Peak RSS, not only the final array sizes, is
the acceptance metric.

## 2026-07-20 local smokes (not an SLA)

One local capacity smoke used a `100 x 100` 2D quad grid with XY spacing
`1.0`, extruded from `z=0` to `z=1000` with Z spacing `1.0`.  The observed
case was:

| Metric | Observed value |
| --- | ---: |
| 2D nodes | 10,201 |
| 2D quads | 10,000 |
| Final 3D nodes | 10,211,201 |
| Final hexahedra | 10,000,000 |
| `mesh_checkerboard()` | 0.005 s |
| `build()` | 0.303 s |
| Peak RSS | 974.5 MiB |

This is a local smoke result, not a latency or memory SLA.  Timings and peak
RSS depend on CPU, memory subsystem, NumPy/Python build, OS, allocator, and
concurrent machine load.  Re-run the production geometry and layer stack on
the deployment hardware; representative geometry is more useful than this
regular-grid microbenchmark for release acceptance.

The same final size was also exercised as 1,000 consecutive full-domain
slabs, each with one Z subdivision.  This exercises layer traversal and exact
capacity reservation rather than a single call with 1,000 subdivisions:

| Metric | Observed value |
| --- | ---: |
| Final 3D nodes | 10,211,201 |
| Final hexahedra | 10,000,000 |
| `build()` | 2.481 s |
| Peak RSS | 938.7 MiB |
| Node/hex capacity | exact final counts |

The exact-reservation result applies only to the proven simple full-domain
BOX schema described above.  Complex areas, holes, metals, or independent
bodies still use the general classification and capacity-growth path and
must be measured with representative production inputs.

Capacity counts alone are not sufficient permission to allocate: every
full-domain slab first passes the same float64 Z-plane representability proof
used by drag.  A deterministically invalid large-Z subdivision therefore
fails before the predicted final arrays are reserved.

## Benchmark commands

Run deterministic complexity regressions:

```bash
python -m pytest tests/test_performance_regression.py -q
```

Show the slowest tests without introducing brittle pass/fail timing limits:

```bash
python -m pytest tests/test_performance_regression.py -q --durations=10
```

On macOS, record peak resident memory for the focused suite:

```bash
/usr/bin/time -l python -m pytest tests/test_performance_regression.py -q
```

Before a production-scale release, run the real representative geometry under
the same command and record: geometry/feature-line count, 2D node and quad
counts, z interval/subdivision counts, final 3D node and hex counts, elapsed
time, and peak RSS.  Compare those values to the previous release rather than
using a machine-independent wall-clock threshold.
