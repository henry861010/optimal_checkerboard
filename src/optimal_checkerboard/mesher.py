"""High-level 2.5D checkerboard mesher interface.

Overview
========
This module is the main client-facing entry point for the 2.5D meshing
workflow used by rectangular advanced-packaging geometries.  The algorithm is
designed for layouts whose xy patterns are axis-aligned rectangles or
orthogonal polygons.  Circular, triangular, and diagonal features are
intentionally out of scope.

The mesher avoids building and organizing the full 3D mesh at once.  Instead,
it projects all pattern edges onto the xy plane, builds a reusable 2D
checkerboard mesh, and lets the caller drag that 2D mesh through z layers.
At each pattern z event, selected checkerboard rail nodes can be snapped back
to the true pattern coordinates before the next extrusion step.

Key Concepts
============
pattern line
    A vertical or horizontal edge extracted from input faces.  A vertical line
    has fixed x and varying y.  A horizontal line has fixed y and varying x.
    Internally each line stores its xy projection plus the z interval where
    that projected edge is active.

shared rail
    A checkerboard grid line that may represent several nearby pattern lines.
    For example, true x pattern lines at x=1.0 and x=1.5 can share one
    checkerboard rail at x=1.25 when their same-z spans do not overlap.

snap rule
    A compact instruction that tells the drag workflow how to move nodes from
    a shared rail back to a true pattern coordinate at the feature bottom z.
    A rule stores axis, rail id, z, target coordinate, and the span to modify.

rail node index
    A sorted node lookup table built after 2D mesh creation.  It allows snap
    rules to find rail nodes with binary search instead of scanning all 2D
    node coordinates.

Workflow
========
1. Create an :class:`OptimalMesh25D` instance.
2. Call :meth:`OptimalMesh25D.set_pattern` with BOX, POLYGON, or LINE faces.
3. Call :meth:`OptimalMesh25D.mesh_checkerboard_box` or
   :meth:`OptimalMesh25D.mesh_assignment`.
4. During 3D drag, call :meth:`OptimalMesh25D.apply_snap_rules_at_z` whenever
   the extrusion reaches a pattern z value.

Minimal Example
===============
    mesher = OptimalMesh25D()
    mesher.set_pattern(faces, element_size=10.0, ratio=0.1)
    mesh2d = mesher.mesh_checkerboard_box([xmin, ymin, xmax, ymax])

    for z_value in sorted(mesher.get_snap_rules()):
        mesher.apply_snap_rules_at_z(z_value)
        # Drag the adjusted mesh2d to the next pattern z in client code.

Input Face Format
=================
BOX faces use a flattened rectangular bound:

    {"type": "BOX", "dim": [x1, y1, x2, y2],
     "bottom_z": z0, "top_z": z1}

POLYGON faces use one closed orthogonal polyline:

    {"type": "POLYGON", "dim": [[x, y], ...],
     "bottom_z": z0, "top_z": z1}

LINE faces use one horizontal or vertical segment:

    {"type": "LINE", "dim": [x1, y1, x2, y2],
     "bottom_z": z0, "top_z": z1}

Important Invariants
====================
- Input feature edges must be horizontal or vertical in xy.
- Every face must declare ``bottom_z`` and ``top_z``.
- Pattern lines whose active z intervals overlap cannot share a rail across
  overlapping or touching xy spans if they would snap to different target
  coordinates.
- Pattern lines with disjoint active z intervals may share a rail.
- The 2D mesh topology is expected to remain compatible with the rail ids
  after snapping.  Node coordinates may move, but the rail index still points
  to the same structural node ids.

Performance Notes
=================
The intended scale is large extruded 3D meshes backed by a moderate 2D mesh.
The expensive search work is pushed into preprocessing:

- shared rails reduce the number of checkerboard grid lines;
- ``rail_node_index`` stores node references sorted by the span axis;
- snap execution is approximately
  O(number_of_rules_at_z * log(nodes_on_rail) + touched_nodes).

Maintenance Notes
=================
Most grouping behavior lives in ``optimal_checkerboard.algorithms``.  This
file should stay focused on orchestration, state ownership, and the public API.
When changing rail compatibility rules, update ``rail_builder.py`` and the
tests that cover same-z overlap, same-z non-overlap, and cross-z sharing.
When changing mesh topology or node ordering, update
``_build_rail_node_index`` and the snap-rule tests together.
"""

from dataclasses import dataclass

import numpy as np

from optimal_checkerboard.algorithms.feature_lines import _get_feature_lines
from optimal_checkerboard.mesh import checkerboard_mesh_box


@dataclass
class Mesh2D:
    """Store public 2D mesh node coordinates and quadrilateral connectivity.

    Attributes:
        nodes: ``(n_nodes, 2+)`` array.  The optional z column is usually zero
            for the base 2D mesh and can be populated by client extrusion code.
        elements: ``(n_elements, 4)`` array of quadrilateral node ids.
    """

    nodes: np.ndarray = None
    elements: np.ndarray = None


class OptimalMesh25D:
    """Build and adjust a shared-rail checkerboard mesh for 2.5D extrusion.

    Public State:
        x_list: Shared x rail coordinates used by the checkerboard mesh.
        y_list: Shared y rail coordinates used by the checkerboard mesh.
        rails: Serialized rail metadata grouped by axis.
        snap_rules_by_z: Mapping from z value to snap rules.
        rail_node_index: Node ids on each rail, sorted by span coordinate.
        mesh2d: The generated or assigned 2D mesh.

    Typical users should call the public methods only.  The private helpers
    are intentionally small because future meshing variants may need to swap
    mesh generation, rail indexing, or snap execution independently.
    """

    def __init__(self):
        """Initialize an empty mesher state."""
        self.element_size = None
        self.faces = None

        self.group_lines_v = None
        self.group_lines_h = None
        self.x_list = None
        self.y_list = None
        self.mesh_x_list = None
        self.mesh_y_list = None

        self.rails = None
        self.snap_rules_by_z = None
        self.rail_reference_index = None
        self.rail_node_index = None
        self.node_axis_rail_ids = None

        self.mesh2d = None
        self.x_pattern_nodes = None
        self.y_pattern_nodes = None

    def set_pattern(self, faces, element_size, ratio=0.1):
        """Extract shared rails and snap rules from pattern faces.

        Args:
            faces: BOX, POLYGON, or LINE face dictionaries.  See the module
                docstring for accepted formats.
            element_size: Preferred checkerboard element size.  This value is
                also used with ``ratio`` to derive the rail merge tolerance.
            ratio: Fraction of ``element_size`` used as the merge tolerance for
                nearby pattern lines.

        Returns:
            A tuple ``(group_lines_v, group_lines_h, x_list, y_list)`` for
            compatibility with earlier client code.

        Side Effects:
            Populates rail metadata, snap rules, and checkerboard rail lists.
        """
        self.element_size = float(element_size)
        self.faces = faces

        merge_tol = ratio * self.element_size
        (
            self.group_lines_v,
            self.group_lines_h,
            self.x_list,
            self.y_list,
            self.snap_rules_by_z,
            self.rails,
        ) = _get_feature_lines(self.faces, merge_tol, return_details=True)

        return self.group_lines_v, self.group_lines_h, self.x_list, self.y_list

    def mesh_checkerboard_box(self, dim=None):
        """Generate a 2D checkerboard mesh from shared rails.

        Args:
            dim: Optional domain bounds.  Use ``[xmin, ymin, xmax, ymax]`` or
                ``[xmin, ymin, zmin, xmax, ymax, zmax]``.  Bounds are inserted
                into the mesh coordinate lists so the checkerboard covers the
                full domain.

        Returns:
            A :class:`Mesh2D` instance containing nodes and elements.
        """
        self._check_pattern_ready()

        self.mesh_x_list, self.mesh_y_list = self._mesh_lists_with_bounds(dim)
        nodes, elements = checkerboard_mesh_box(
            self.mesh_x_list,
            self.mesh_y_list,
            self.element_size,
        )

        self.mesh2d = Mesh2D(nodes=nodes, elements=elements)
        self._validate_mesh2d_arrays()
        self._build_rail_node_index()
        return self.mesh2d

    def mesh_assignment(self, mesh2d):
        """Assign a user-provided 2D mesh and build rail lookup tables.

        Args:
            mesh2d: Object with ``nodes`` and ``elements`` attributes.  Nodes
                must include coordinates that match every shared rail in
                ``x_list`` and ``y_list`` within tolerance.

        Returns:
            The assigned mesh object.
        """
        self._check_pattern_ready()

        self.mesh2d = mesh2d
        self.mesh_x_list = np.asarray(self.x_list, dtype=np.float64)
        self.mesh_y_list = np.asarray(self.y_list, dtype=np.float64)
        self._validate_mesh2d_arrays()
        self._build_rail_node_index()
        return self.mesh2d

    def get_snap_rules(self, z=None, eps=1e-6):
        """Return all snap rules or the rules matching a specific z value.

        Args:
            z: Optional z coordinate.  If omitted, returns the full mapping.
            eps: Matching tolerance for z lookup.

        Returns:
            Either ``snap_rules_by_z`` or a list of rules for one z event.
        """
        if self.snap_rules_by_z is None:
            return {} if z is None else []

        if z is None:
            return self.snap_rules_by_z

        z = float(z)
        if z in self.snap_rules_by_z:
            return self.snap_rules_by_z[z]

        for z_key, rules in self.snap_rules_by_z.items():
            if abs(float(z_key) - z) <= eps:
                return rules
        return []

    def apply_snap_rules_at_z(self, z, nodes=None, eps=1e-6):
        """Move rail nodes to true pattern coordinates.

        Args:
            z: Pattern z value whose snap rules should be applied.
            nodes: Optional ``(n, 2+)`` node array to mutate. If omitted,
                mutates ``self.mesh2d.nodes``.
            eps: Tolerance used for z matching and span selection.

        Returns:
            Number of mesh nodes touched by all applied rules.

        Notes:
            This method intentionally mutates node coordinates in place.  The
            caller can apply the rules once at a pattern z, then drag the
            adjusted mesh nodes to the next z interval.
        """
        if self.rail_node_index is None:
            raise RuntimeError(
                "Error: mesh_checkerboard_box or mesh_assignment is not "
                "performed"
            )

        if nodes is None:
            nodes = self.mesh2d.nodes

        nodes = np.asarray(nodes)
        if nodes.ndim != 2 or nodes.shape[1] < 2:
            raise ValueError("nodes must have shape (n, 2+)")
        if not np.issubdtype(nodes.dtype, np.floating):
            raise ValueError("nodes must contain floating point coordinates")

        rules = self.get_snap_rules(z, eps=eps)
        rules_by_axis = self._rules_by_axis_and_rail(rules)
        target_values = {
            "x": np.zeros(len(nodes), dtype=np.float64),
            "y": np.zeros(len(nodes), dtype=np.float64),
        }
        target_masks = {
            "x": np.zeros(len(nodes), dtype=bool),
            "y": np.zeros(len(nodes), dtype=bool),
        }

        touched = 0
        for rule in rules:
            node_ids = self._snap_rule_node_ids(
                rule,
                rules_by_axis,
                eps=eps,
            )
            axis = rule["axis"]
            target_values[axis][node_ids] = rule["target_coord"]
            target_masks[axis][node_ids] = True
            touched += int(len(node_ids))

        for axis, coord_axis in (("x", 0), ("y", 1)):
            mask = target_masks[axis]
            nodes[mask, coord_axis] = target_values[axis][mask]
        return touched

    def _apply_snap_rule(self, nodes, rule, eps=1e-6):
        """Apply one snap rule to a sorted rail-node span."""
        node_ids = self._span_node_ids(rule, eps=eps)

        coord_axis = 0 if rule["axis"] == "x" else 1
        nodes[node_ids, coord_axis] = rule["target_coord"]
        return int(len(node_ids))

    def _span_node_ids(self, rule, eps=1e-6):
        """Return nodes on a rail whose structural span lies in one rule."""
        axis = rule["axis"]
        rail_id = rule["rail_id"]

        rail_index = self.rail_node_index[axis][rail_id]
        sorted_span_values = rail_index["span_values"]
        sorted_node_ids = rail_index["node_ids"]

        lo = np.searchsorted(
            sorted_span_values,
            rule["span_min"] - eps,
            side="left",
        )
        hi = np.searchsorted(
            sorted_span_values,
            rule["span_max"] + eps,
            side="right",
        )
        return sorted_node_ids[lo:hi]

    def _snap_rule_node_ids(self, rule, rules_by_axis, eps=1e-6):
        """Return all nodes affected by a rule, including snapped corners."""
        node_ids = self._span_node_ids(rule, eps=eps)
        corner_node_ids = self._coupled_corner_node_ids(
            rule,
            rules_by_axis,
            eps=eps,
        )

        if not len(corner_node_ids):
            return node_ids
        if not len(node_ids):
            return np.unique(corner_node_ids)
        return np.unique(np.concatenate((node_ids, corner_node_ids)))

    def _coupled_corner_node_ids(self, rule, rules_by_axis, eps=1e-6):
        """Return corner nodes whose opposite rail snaps into this span."""
        if self.node_axis_rail_ids is None:
            return np.empty(0, dtype=np.intp)

        axis = rule["axis"]
        opposite_axis = "y" if axis == "x" else "x"
        opposite_rule_lists = rules_by_axis[opposite_axis]
        if not opposite_rule_lists:
            return np.empty(0, dtype=np.intp)

        rail_node_ids = self.rail_node_index[axis][rule["rail_id"]]["node_ids"]
        opposite_rail_ids = self.node_axis_rail_ids[opposite_axis][
            rail_node_ids
        ]
        matching_node_ids = []

        for opposite_rail_id in np.unique(opposite_rail_ids):
            if opposite_rail_id < 0:
                continue

            for opposite_rule in opposite_rule_lists.get(
                int(opposite_rail_id),
                (),
            ):
                if not self._rule_targets_cross(rule, opposite_rule, eps=eps):
                    continue

                matching_node_ids.append(
                    rail_node_ids[opposite_rail_ids == opposite_rail_id]
                )
                break

        if not matching_node_ids:
            return np.empty(0, dtype=np.intp)
        return np.concatenate(matching_node_ids)

    def _rule_targets_cross(self, rule, opposite_rule, eps=1e-6):
        """Return whether two perpendicular snap rules meet at a corner."""
        return (
            rule["span_min"] - eps
            <= opposite_rule["target_coord"]
            <= rule["span_max"] + eps
            and opposite_rule["span_min"] - eps
            <= rule["target_coord"]
            <= opposite_rule["span_max"] + eps
        )

    def _rules_by_axis_and_rail(self, rules):
        """Group snap rules by axis and rail id for same-z corner lookup."""
        rules_by_axis = {"x": {}, "y": {}}
        for rule in rules:
            rules_by_axis[rule["axis"]].setdefault(
                rule["rail_id"],
                [],
            ).append(rule)
        return rules_by_axis

    def _check_pattern_ready(self):
        """Raise an error if pattern preprocessing has not been completed."""
        if (
            self.x_list is None
            or self.y_list is None
            or self.element_size is None
        ):
            raise RuntimeError(
                "Error: OptimalMesh25D.set_pattern is not performed"
            )

    def _mesh_lists_with_bounds(self, dim):
        """Return rail coordinate lists with optional box bounds."""
        x_values = list(self.x_list)
        y_values = list(self.y_list)

        if dim is not None:
            if len(dim) == 4:
                xmin, ymin, xmax, ymax = dim
            elif len(dim) == 6:
                xmin, ymin, _, xmax, ymax, _ = dim
            else:
                raise ValueError(
                    "dim must be [xmin, ymin, xmax, ymax] or "
                    "[xmin, ymin, zmin, xmax, ymax, zmax]"
                )

            x_values.extend([xmin, xmax])
            y_values.extend([ymin, ymax])

        return self._unique_sorted(x_values), self._unique_sorted(y_values)

    def _unique_sorted(self, values, eps=1e-9):
        """Return sorted float values with near-duplicates removed."""
        values = sorted(float(value) for value in values)
        if not values:
            return np.asarray([], dtype=np.float64)

        unique_values = [values[0]]
        for value in values[1:]:
            if abs(value - unique_values[-1]) > eps:
                unique_values.append(value)
        return np.asarray(unique_values, dtype=np.float64)

    def _validate_mesh2d_arrays(self):
        """Validate and normalize the assigned public mesh arrays."""
        nodes = np.asarray(self.mesh2d.nodes)
        elements = np.asarray(self.mesh2d.elements)

        if nodes.ndim != 2 or nodes.shape[1] < 2:
            raise ValueError("mesh2d.nodes must have shape (n, 2+)")
        if not np.issubdtype(nodes.dtype, np.floating):
            nodes = nodes.astype(np.float64)

        if elements.ndim != 2 or elements.shape[1] != 4:
            raise ValueError("mesh2d.elements must have shape (m, 4)")
        if not np.issubdtype(elements.dtype, np.integer):
            raise ValueError("mesh2d.elements must contain integer node ids")

        if elements.size:
            min_node_id = int(elements.min())
            max_node_id = int(elements.max())
            if min_node_id < 0 or max_node_id >= len(nodes):
                raise ValueError(
                    "mesh2d.elements contain node ids outside mesh2d.nodes"
                )

        self.mesh2d.nodes = nodes
        self.mesh2d.elements = elements
        return nodes, elements

    def _build_rail_node_index(self, eps=1e-6):
        """Build sorted node indices for snap span queries."""
        nodes, elements = self._validate_mesh2d_arrays()
        self.rail_node_index = {
            "x": self._build_axis_node_index(
                nodes,
                elements,
                self.rails["x"],
                coord_axis=0,
                span_axis=1,
                eps=eps,
            ),
            "y": self._build_axis_node_index(
                nodes,
                elements,
                self.rails["y"],
                coord_axis=1,
                span_axis=0,
                eps=eps,
            ),
        }
        self.node_axis_rail_ids = self._build_node_axis_rail_ids(len(nodes))
        self.rail_reference_index = self.rail_node_index

    def _build_node_axis_rail_ids(self, node_count):
        """Return each node's structural rail id for both axes."""
        node_axis_rail_ids = {
            "x": np.full(node_count, -1, dtype=np.intp),
            "y": np.full(node_count, -1, dtype=np.intp),
        }

        for axis, rail_indices in self.rail_node_index.items():
            for rail_id, rail_index in enumerate(rail_indices):
                node_axis_rail_ids[axis][rail_index["node_ids"]] = rail_id

        return node_axis_rail_ids

    def _build_axis_node_index(
        self,
        nodes,
        elements,
        rails,
        coord_axis,
        span_axis,
        eps,
    ):
        """Index active nodes on one axis with one global coordinate sort."""
        if elements.size:
            active_node_ids = np.unique(elements.ravel()).astype(np.intp)
        else:
            active_node_ids = np.empty(0, dtype=np.intp)

        coord_values = nodes[active_node_ids, coord_axis]
        span_values = nodes[active_node_ids, span_axis]
        coord_order = np.argsort(coord_values, kind="mergesort")
        sorted_coord_values = coord_values[coord_order]

        rail_indices = []
        for rail in rails:
            lo = np.searchsorted(
                sorted_coord_values,
                rail["coord"] - eps,
                side="left",
            )
            hi = np.searchsorted(
                sorted_coord_values,
                rail["coord"] + eps,
                side="right",
            )

            node_refs = coord_order[lo:hi]
            span_order = np.argsort(span_values[node_refs], kind="mergesort")
            node_refs = node_refs[span_order]

            rail_indices.append(
                {
                    "node_ids": active_node_ids[node_refs],
                    "span_values": span_values[node_refs],
                }
            )

        return rail_indices
