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

shared rail
    A checkerboard grid line that may represent several nearby pattern lines.
    For example, true x pattern lines at x=1.0 and x=1.5 can share one
    checkerboard rail at x=1.25 when their same-z spans do not overlap.

snap rule
    A compact instruction that tells the drag workflow how to move nodes from
    a shared rail back to a true pattern coordinate at one z event.  A rule
    stores axis, rail id, z, target coordinate, and the span to modify.

rail reference index
    A sorted element-corner lookup table built after 2D mesh creation.  It
    allows snap rules to find internal references with binary search instead
    of scanning all 2D element coordinates.

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

    {"type": "BOX", "dim": [x1, y1, z1, x2, y2, z2]}

Only the lower z value is used when extracting the face edges.  POLYGON faces
use one or more closed orthogonal polylines:

    {"type": "POLYGON", "dim": [[[x, y, z], ...], ...]}

LINE faces use one horizontal or vertical 3D segment on a single z plane, so
``z1`` and ``z2`` must be equal:

    {"type": "LINE", "dim": [[x1, y1, z1], [x2, y2, z2]]}

Important Invariants
====================
- Input feature edges must be horizontal or vertical in xy.
- Each input line must lie on exactly one z plane.
- Same-z pattern lines that overlap or touch along their span cannot share a
  rail if they would snap to different target coordinates.
- Cross-z pattern lines may share a rail, because snap rules are applied only
  at one z event at a time.
- The 2D mesh topology is expected to remain compatible with the rail ids
  after snapping.  Node coordinates may move, but the rail index still points
  to the same structural node ids.

Performance Notes
=================
The intended scale is large extruded 3D meshes backed by a moderate 2D mesh.
The expensive search work is pushed into preprocessing:

- shared rails reduce the number of checkerboard grid lines;
- ``rail_reference_index`` stores element-corner references sorted by the span
  axis;
- snap execution is approximately
  O(number_of_rules_at_z * log(references_on_rail) + touched_references).

Maintenance Notes
=================
Most grouping behavior lives in ``optimal_checkerboard.algorithms``.  This
file should stay focused on orchestration, state ownership, and the public API.
When changing rail compatibility rules, update ``rail_builder.py`` and the
tests that cover same-z overlap, same-z non-overlap, and cross-z sharing.
When changing mesh topology or node ordering, update
``_build_rail_reference_index`` and the snap-rule tests together.
"""

from dataclasses import dataclass

import numpy as np

from optimal_checkerboard.algorithms.feature_lines import _get_feature_lines
from optimal_checkerboard.mesh import checkerboard_mesh_box


ELEMENT_2D_COMP_ID = 0
ELEMENT_2D_VOLUMN = 1
ELEMENT_2D_VOLUME = ELEMENT_2D_VOLUMN
ELEMENT_2D_NODE1_X = 2
ELEMENT_2D_NODE1_Y = 3
ELEMENT_2D_NODE2_X = 4
ELEMENT_2D_NODE2_Y = 5
ELEMENT_2D_NODE3_X = 6
ELEMENT_2D_NODE3_Y = 7
ELEMENT_2D_NODE4_X = 8
ELEMENT_2D_NODE4_Y = 9

ELEMENT_2D_X_COLUMNS = np.asarray(
    [
        ELEMENT_2D_NODE1_X,
        ELEMENT_2D_NODE2_X,
        ELEMENT_2D_NODE3_X,
        ELEMENT_2D_NODE4_X,
    ],
    dtype=np.intp,
)
ELEMENT_2D_Y_COLUMNS = np.asarray(
    [
        ELEMENT_2D_NODE1_Y,
        ELEMENT_2D_NODE2_Y,
        ELEMENT_2D_NODE3_Y,
        ELEMENT_2D_NODE4_Y,
    ],
    dtype=np.intp,
)


@dataclass
class Mesh2D:
    """Store public 2D mesh arrays and the drag-time internal element table.

    Attributes:
        nodes: ``(n_nodes, 3)`` array.  The z column is usually zero for the
            base 2D mesh and can be populated by client extrusion code.
        elements: ``(n_elements, 4)`` array of quadrilateral node ids.
        element_internal: ``(n_elements, 10)`` drag-time table.  Columns are
            component id, 2D area, and the four element corner xy coordinates.
    """

    nodes: np.ndarray = None
    elements: np.ndarray = None
    element_internal: np.ndarray = None


class OptimalMesh25D:
    """Build and adjust a shared-rail checkerboard mesh for 2.5D extrusion.

    Public State:
        x_list: Shared x rail coordinates used by the checkerboard mesh.
        y_list: Shared y rail coordinates used by the checkerboard mesh.
        rails: Serialized rail metadata grouped by axis.
        snap_rules_by_z: Mapping from z value to snap rules.
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
            A :class:`Mesh2D` instance containing nodes, elements, and
            ``element_internal``.
        """
        self._check_pattern_ready()

        self.mesh_x_list, self.mesh_y_list = self._mesh_lists_with_bounds(dim)
        nodes, elements = checkerboard_mesh_box(
            self.mesh_x_list,
            self.mesh_y_list,
            self.element_size,
        )

        self.mesh2d = Mesh2D(nodes=nodes, elements=elements)
        self._build_element_internal()
        self._build_rail_reference_index()
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
        self._build_element_internal()
        self._build_rail_reference_index()
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

    def apply_snap_rules_at_z(self, z, element_internal=None, eps=1e-6):
        """Move element corner references to true pattern coordinates.

        Args:
            z: Pattern z value whose snap rules should be applied.
            element_internal: Optional ``(m, 10)`` internal table to mutate.
                If omitted, mutates ``self.mesh2d.element_internal``.
            eps: Tolerance used for z matching and span selection.

        Returns:
            Number of element corner references touched by all applied rules.

        Notes:
            This method intentionally mutates element-corner coordinates in
            place.  The caller can apply the rules once at a pattern z, then
            drag the adjusted internal table to the next z interval.
        """
        if self.rail_reference_index is None:
            raise RuntimeError(
                "Error: mesh_checkerboard_box or mesh_assignment is not "
                "performed"
            )

        if element_internal is None:
            element_internal = self.mesh2d.element_internal

        element_internal = np.asarray(element_internal)
        if element_internal.ndim != 2 or element_internal.shape[1] != 10:
            raise ValueError("element_internal must have shape (m, 10)")

        touched = 0
        for rule in self.get_snap_rules(z, eps=eps):
            touched += self._apply_snap_rule(element_internal, rule, eps=eps)
        return touched

    def _apply_snap_rule(self, element_internal, rule, eps=1e-6):
        """Apply one snap rule to a sorted element-corner reference span."""
        axis = rule["axis"]
        rail_id = rule["rail_id"]

        rail_index = self.rail_reference_index[axis][rail_id]
        sorted_span_values = rail_index["span_values"]
        sorted_element_ids = rail_index["element_ids"]
        sorted_coord_columns = rail_index["coord_columns"]

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
        element_ids = sorted_element_ids[lo:hi]
        coord_columns = sorted_coord_columns[lo:hi]

        element_internal[element_ids, coord_columns] = rule["target_coord"]
        return int(len(element_ids))

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

    def _build_element_internal(self):
        """Convert public nodes/elements into the drag-time internal table."""
        element_internal = getattr(self.mesh2d, "element_internal", None)
        if element_internal is not None:
            element_internal = np.asarray(element_internal)
            if element_internal.ndim != 2 or element_internal.shape[1] != 10:
                raise ValueError(
                    "mesh2d.element_internal must have shape (m, 10)"
                )
            self.mesh2d.element_internal = element_internal
            return

        nodes = np.asarray(self.mesh2d.nodes)
        elements = np.asarray(self.mesh2d.elements, dtype=np.intp)
        if nodes.ndim != 2 or nodes.shape[1] < 2:
            raise ValueError("mesh2d.nodes must have shape (n, 3)")
        if elements.ndim != 2 or elements.shape[1] != 4:
            raise ValueError("mesh2d.elements must have shape (m, 4)")

        corner_xy = nodes[elements, :2]
        element_count = elements.shape[0]
        element_internal = np.empty((element_count, 10), dtype=np.float32)
        element_internal[:, ELEMENT_2D_COMP_ID] = 0.0
        element_internal[:, ELEMENT_2D_VOLUMN] = _quad_area_xy(corner_xy)
        element_internal[:, 2:10] = corner_xy.reshape(element_count, 8)
        self.mesh2d.element_internal = element_internal

    def _build_rail_reference_index(self, eps=1e-6):
        """Build sorted element-reference indices for snap span queries."""
        element_internal = self.mesh2d.element_internal
        self.rail_reference_index = {
            "x": self._build_axis_reference_index(
                element_internal,
                self.rails["x"],
                ELEMENT_2D_X_COLUMNS,
                ELEMENT_2D_Y_COLUMNS,
                eps,
            ),
            "y": self._build_axis_reference_index(
                element_internal,
                self.rails["y"],
                ELEMENT_2D_Y_COLUMNS,
                ELEMENT_2D_X_COLUMNS,
                eps,
            ),
        }
        self.rail_node_index = self.rail_reference_index

    def _build_axis_reference_index(
        self,
        element_internal,
        rails,
        coord_columns,
        span_columns,
        eps,
    ):
        """Index all references on one axis with one global coordinate sort."""
        coord_values = element_internal[:, coord_columns].ravel()
        span_values = element_internal[:, span_columns].ravel()
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

            flat_refs = coord_order[lo:hi]
            span_order = np.argsort(span_values[flat_refs], kind="mergesort")
            flat_refs = flat_refs[span_order]

            rail_indices.append(
                {
                    "element_ids": (flat_refs // 4).astype(np.intp),
                    "coord_columns": coord_columns[flat_refs % 4],
                    "span_values": span_values[flat_refs],
                }
            )

        return rail_indices


def _quad_area_xy(corner_xy):
    """Return vectorized absolute xy area for quadrilateral corner arrays."""
    x_values = corner_xy[:, :, 0]
    y_values = corner_xy[:, :, 1]
    next_x_values = np.roll(x_values, -1, axis=1)
    next_y_values = np.roll(y_values, -1, axis=1)
    twice_area = np.sum(
        x_values * next_y_values - y_values * next_x_values,
        axis=1,
    )
    return 0.5 * np.abs(twice_area)
