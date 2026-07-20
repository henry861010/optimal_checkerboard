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
    Restore rules are indexed by feature top z, staged at that z, and applied
    only when the next higher z layer is processed.

rail node index
    A sorted node lookup table built after 2D mesh creation.  It allows snap
    rules to find rail nodes with binary search instead of scanning all 2D
    node coordinates.

Workflow
========
1. Create an :class:`OptimalMesh25D` instance.
2. Call :meth:`OptimalMesh25D.set_pattern_obj` with a geometry ``Obj``.
3. For custom 2D meshing, optionally call
   :meth:`OptimalMesh25D.get_snap_faces` and build the mesh from those
   shared-rail default faces.
4. Call :meth:`OptimalMesh25D.mesh_checkerboard` or
   :meth:`OptimalMesh25D.mesh_assignment`.
5. During 3D drag, call :meth:`OptimalMesh25D.apply_snap_rules_at_z` whenever
   the extrusion reaches a pattern z value.

Minimal Example
===============
    mesher = OptimalMesh25D()
    mesher.set_pattern_obj(obj, element_size=10.0, ratio=0.1)
    mesh2d = mesher.mesh_checkerboard()

    z_events = sorted(
        set(mesher.get_snap_rules()) | set(mesher.get_restore_rules())
    )
    for z_value in z_events:
        mesher.apply_snap_rules_at_z(z_value)
        # Drag the adjusted mesh2d to the next pattern z in client code.

Private Raw Face Format
=======================
The private :meth:`OptimalMesh25D._set_pattern` helper accepts raw face
dictionaries for tests and debug scripts.  Normal client code should prefer
:meth:`OptimalMesh25D.set_pattern_obj`.

BOX faces use a flattened rectangular bound:

    {"type": "BOX", "dim": [x1, y1, x2, y2],
     "bottom_z": z0, "top_z": z1}

POLYGON faces use one or more closed orthogonal loops.  Clockwise loops are
hulls; counter-clockwise loops are holes:

    {"type": "POLYGON", "dim": [[[x, y], ...], ...],
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
from typing import TYPE_CHECKING
import warnings

import numpy as np

from optimal_checkerboard.algorithms.feature_lines import _get_feature_lines
from optimal_checkerboard.algorithms.drag import Dragger
from optimal_checkerboard.algorithms.snap_faces import snap_faces_to_shared_rails
from optimal_checkerboard.mesh import (
    generate_checkerboard_mesh,
    mesh_domain_from_obj,
    normalize_mesh_domain,
)

if TYPE_CHECKING:
    from optimal_checkerboard.data_structure.geometry import Obj


_PATTERN_FACE_TYPES = {"BOX", "LINE", "POLYGON"}


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
        restore_rules_by_z: Mapping from top z to delayed restore rules.
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
        self.mesh_domain = None

        self.group_lines_v = None
        self.group_lines_h = None
        self.x_list = None
        self.y_list = None
        self.mesh_x_list = None
        self.mesh_y_list = None

        self.rails = None
        self.snap_rules_by_z = None
        self.restore_rules_by_z = None
        self._restore_rule_buffer = []
        self._restore_buffer_source_z = None
        self.rail_reference_index = None
        self.rail_node_index = None
        self.node_axis_rail_ids = None

        self.mesh2d = None
        self.dragger = None
        self.x_pattern_nodes = None
        self.y_pattern_nodes = None

    def _set_pattern(self, faces, element_size, ratio=0.1, mesh_domain=None):
        """Extract shared rails plus snap and restore rules from raw faces.

        Args:
            faces: BOX, POLYGON, or LINE face dictionaries.  See the module
                docstring for accepted formats.  This raw schema is intended
                for tests and debug scripts; client code should use
                :meth:`set_pattern_obj`.
            element_size: Preferred checkerboard element size.  This value is
                also used with ``ratio`` to derive the rail merge tolerance.
            ratio: Fraction of ``element_size`` used as the merge tolerance for
                nearby pattern lines.
            mesh_domain: Optional normalized root footprint domain for private
                raw-face callers that still want generated checkerboard meshes.

        Returns:
            A tuple ``(group_lines_v, group_lines_h, x_list, y_list)`` for
            compatibility with earlier client code.

        Side Effects:
            Populates rail metadata, snap/restore rules, and checkerboard
            rail lists.
        """
        self.element_size = float(element_size)
        self.faces = faces
        self.mesh_domain = normalize_mesh_domain(mesh_domain)

        merge_tol = ratio * self.element_size
        (
            self.group_lines_v,
            self.group_lines_h,
            self.x_list,
            self.y_list,
            self.snap_rules_by_z,
            self.restore_rules_by_z,
            self.rails,
        ) = _get_feature_lines(self.faces, merge_tol, return_details=True)
        self.reset_snap_state()

        return self.group_lines_v, self.group_lines_h, self.x_list, self.y_list

    def set_pattern_obj(self, obj: "Obj", element_size, ratio=0.1):
        """Extract pattern faces from an ``Obj`` hierarchy.

        The conversion uses object, metal range/hole, mesh-line, mesh-face, and
        child-object boundaries as pattern faces.  Metals without explicit
        ranges or holes do not add new pattern faces because they inherit their
        parent object's footprint.  CYLINDER faces are ignored with a warning
        because the pattern mesher only consumes orthogonal edges, but a root
        CYLINDER footprint is still saved as the mesh domain for future 2D
        mesh generation.
        """
        obj.set_position_abs(0, 0, 0)
        mesh_domain = mesh_domain_from_obj(obj)
        faces = self._pattern_faces_from_obj(obj)
        return self._set_pattern(
            faces,
            element_size=element_size,
            ratio=ratio,
            mesh_domain=mesh_domain,
        )

    def _pattern_faces_from_obj(self, obj: "Obj"):
        """Return raw pattern face dictionaries from one absolute ``Obj`` tree."""
        faces = []

        self._append_pattern_face(
            faces,
            obj.face.type,
            obj.face.dim_abs,
            obj.z_abs,
            None if obj.z_abs is None else obj.z_abs + obj.thk,
        )

        for metal in obj.metals:
            for face_range in metal.ranges + metal.holes:
                self._append_pattern_face(
                    faces,
                    face_range.type,
                    face_range.dim_abs,
                    metal.begin_abs,
                    metal.end_abs,
                )

        for mesh in obj.meshs:
            if mesh.line is not None:
                self._append_pattern_face(
                    faces,
                    "LINE",
                    self._mesh_line_dim(mesh),
                    mesh.begin_abs,
                    mesh.end_abs,
                )
            if mesh.face is not None:
                self._append_pattern_face(
                    faces,
                    mesh.face.type,
                    mesh.face.dim_abs,
                    mesh.begin_abs,
                    mesh.end_abs,
                )

        for child_obj in obj.child_objs:
            faces.extend(self._pattern_faces_from_obj(child_obj))

        return faces

    def _append_pattern_face(self, faces, face_type, dim, bottom_z, top_z):
        """Append one validated raw pattern face dictionary."""
        if face_type == "CYLINDER":
            warnings.warn(
                "CYLINDER faces are ignored during pattern conversion",
                RuntimeWarning,
                stacklevel=2,
            )
            return

        if face_type not in _PATTERN_FACE_TYPES:
            raise ValueError(
                "Unsupported Obj face type for pattern conversion: "
                f"{face_type}. Supported types are BOX, LINE, and POLYGON."
            )
        if dim is None:
            raise ValueError(
                f"{face_type} face has no absolute coordinates. "
                "Call set_position_abs before pattern conversion."
            )
        if bottom_z is None or top_z is None:
            raise ValueError(
                f"{face_type} face has no absolute z interval. "
                "Call set_position_abs before pattern conversion."
            )

        faces.append(
            {
                "type": face_type,
                "dim": dim,
                "bottom_z": bottom_z,
                "top_z": top_z,
            }
        )

    def _mesh_line_dim(self, mesh):
        """Return mesh line absolute coordinates as [x1, y1, x2, y2]."""
        if mesh.line_abs is None or len(mesh.line_abs) != 2:
            raise ValueError(
                "Mesh line has no valid absolute coordinates. "
                "Call set_position_abs before pattern conversion."
            )

        point1, point2 = mesh.line_abs
        if len(point1) < 2 or len(point2) < 2:
            raise ValueError("Mesh line points must contain x and y")
        return [point1[0], point1[1], point2[0], point2[1]]

    def mesh_checkerboard(self):
        """Generate a 2D checkerboard mesh from the root footprint domain.

        ``set_pattern_obj`` resolves the root ``Obj`` footprint into
        ``self.mesh_domain``.  This dispatcher then selects the matching 2D
        mesh generator for that domain type.

        Returns:
            A :class:`Mesh2D` instance containing nodes and elements.
        """
        return self._mesh_checkerboard_for_domain()

    def _mesh_checkerboard_for_domain(self, model_type="Full Model", center_x=None, center_y=None, required_domain_type=None):
        """Generate and index a checkerboard mesh for ``self.mesh_domain``."""
        self._check_pattern_ready()
        self._check_mesh_domain_ready()

        if (
            required_domain_type is not None
            and self.mesh_domain["type"] != required_domain_type
        ):
            raise ValueError(
                f"mesh_checkerboard_{required_domain_type.lower()} requires "
                f"a {required_domain_type} root footprint"
            )

        (
            nodes,
            elements,
            self.mesh_x_list,
            self.mesh_y_list,
        ) = generate_checkerboard_mesh(
            self.mesh_domain,
            self.x_list,
            self.y_list,
            self.element_size,
        )

        self.mesh2d = Mesh2D(nodes=nodes, elements=elements)
        self._validate_mesh2d_arrays()
        self._build_rail_node_index()
        self.reset_snap_state()
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
        self.reset_snap_state()
        return self.mesh2d

    def get_snap_rules(self, z=None, eps=1e-6):
        """Return all snap rules or the rules matching a specific z value.

        Args:
            z: Optional z coordinate.  If omitted, returns the full mapping.
            eps: Matching tolerance for z lookup.

        Returns:
            Either ``snap_rules_by_z`` or a list of rules for one z event.
        """
        return self._get_rules_from_z_map(self.snap_rules_by_z, z, eps=eps)

    def get_restore_rules(self, z=None, eps=1e-6):
        """Return delayed restore rules indexed by their feature top z.

        Restore rules found at z are staged during that call and applied at
        the next higher z passed to :meth:`apply_snap_rules_at_z`.
        """
        return self._get_rules_from_z_map(
            self.restore_rules_by_z,
            z,
            eps=eps,
        )

    def get_snap_faces(self, eps=1e-6):
        """Return pattern faces adjusted to shared-rail default coordinates.

        The returned face dictionaries are deep copies of ``self.faces``.
        Coordinates that participate in snap rules are moved from their true
        pattern coordinate to the shared rail coordinate.  This gives callers a
        rail-compatible face set for generating or validating a custom 2D mesh;
        the original faces and snap rules remain unchanged for later z-layer
        snapping.
        """
        self._check_pattern_ready()
        return snap_faces_to_shared_rails(
            self.faces,
            self.snap_rules_by_z,
            eps=eps,
        )

    def reset_snap_state(self):
        """Clear delayed restore state for a fresh z traversal."""
        self._restore_rule_buffer = []
        self._restore_buffer_source_z = None

    def _get_rules_from_z_map(self, rules_by_z, z=None, eps=1e-6):
        """Return all z-indexed rules or one tolerant z bucket."""
        if rules_by_z is None:
            return {} if z is None else []

        if z is None:
            return rules_by_z

        _, rules = self._get_rules_and_z_from_z_map(rules_by_z, z, eps=eps)
        return rules

    def _get_rules_and_z_from_z_map(self, rules_by_z, z, eps=1e-6):
        """Return the matched z key and rule bucket from a z-indexed map."""
        if rules_by_z is None:
            return None, []

        z = float(z)
        if z in rules_by_z:
            return z, rules_by_z[z]

        for z_key, rules in rules_by_z.items():
            if abs(float(z_key) - z) <= eps:
                return z_key, rules
        return None, []

    def apply_snap_rules_at_z(
        self,
        z,
        nodes=None,
        eps=1e-6,
        return_touched_node_ids=False,
    ):
        """Move rail nodes to true pattern coordinates.

        Args:
            z: Pattern z value whose snap rules should be applied, and whose
                top-z restore rules should be staged for the next higher z.
            nodes: Optional ``(n, 2+)`` node array to mutate. If omitted,
                mutates ``self.mesh2d.nodes``.
            eps: Tolerance used for z matching and span selection.
            return_touched_node_ids: If true, also return unique node ids that
                were touched by the snap operation.

        Returns:
            Number of mesh nodes touched by all applied rules, or a tuple of
            ``(touched_count, touched_node_ids)`` when
            ``return_touched_node_ids`` is true.

        Notes:
            This method intentionally mutates node coordinates in place.  The
            caller can apply the rules once at a pattern z, then drag the
            adjusted mesh nodes to the next z interval.
        """
        if self.rail_node_index is None:
            raise RuntimeError(
                "Error: mesh_checkerboard or "
                "mesh_assignment is not performed"
            )

        if nodes is None:
            nodes = self.mesh2d.nodes

        nodes = np.asarray(nodes)
        if nodes.ndim != 2 or nodes.shape[1] < 2:
            raise ValueError("nodes must have shape (n, 2+)")
        if not np.issubdtype(nodes.dtype, np.floating):
            raise ValueError("nodes must contain floating point coordinates")

        z = float(z)
        restore_rules = self._consume_restore_rule_buffer(z, eps=eps)
        snap_rules = self.get_snap_rules(z, eps=eps)
        touched = self._apply_snap_rule_batch(
            nodes,
            restore_rules + snap_rules,
            eps=eps,
            return_touched_node_ids=return_touched_node_ids,
        )
        self._stage_restore_rules_at_z(z, eps=eps)
        return touched

    def _apply_snap_rule_batch(
        self,
        nodes,
        rules,
        eps=1e-6,
        return_touched_node_ids=False,
    ):
        """Apply same-z restore and snap rules as one atomic update."""
        if not rules:
            if return_touched_node_ids:
                return 0, np.empty(0, dtype=np.intp)
            return 0

        rules_by_axis = self._rules_by_axis_and_rail(rules)
        updates = []
        touched_node_chunks = []
        touched = 0

        for rule in rules:
            node_ids = self._snap_rule_node_ids(
                rule,
                rules_by_axis,
                eps=eps,
            )
            coord_axis = 0 if rule["axis"] == "x" else 1
            updates.append((node_ids, coord_axis, rule["target_coord"]))
            touched += int(len(node_ids))

            if return_touched_node_ids and len(node_ids):
                touched_node_chunks.append(node_ids)

        for node_ids, coord_axis, target_coord in updates:
            nodes[node_ids, coord_axis] = target_coord

        if return_touched_node_ids:
            if touched_node_chunks:
                touched_node_ids = np.unique(np.concatenate(touched_node_chunks))
            else:
                touched_node_ids = np.empty(0, dtype=np.intp)
            return touched, touched_node_ids
        return touched

    def _consume_restore_rule_buffer(self, z, eps=1e-6):
        """Return staged restore rules only after their top-z layer passes."""
        if not self._restore_rule_buffer:
            return []

        source_z = self._restore_buffer_source_z
        if source_z is None or z <= float(source_z) + eps:
            return []

        rules = self._restore_rule_buffer
        self.reset_snap_state()
        return rules

    def _stage_restore_rules_at_z(self, z, eps=1e-6):
        """Stage top-z restore rules for the next higher z layer."""
        source_z, rules = self._get_rules_and_z_from_z_map(
            self.restore_rules_by_z,
            z,
            eps=eps,
        )
        if source_z is None or not rules:
            return

        self._restore_rule_buffer = list(rules)
        self._restore_buffer_source_z = source_z

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
                "Error: OptimalMesh25D.set_pattern_obj or _set_pattern is "
                "not performed"
            )

    def _check_mesh_domain_ready(self):
        """Raise an error if generated meshing has no root boundary."""
        if self.mesh_domain is None:
            raise RuntimeError(
                "Error: root mesh boundary is not available. Use "
                "OptimalMesh25D.set_pattern_obj with an Obj root footprint "
                "before mesh_checkerboard."
            )

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

    def build(self, obj_list, preserve_mesh2d=False):
        """Build a 3D mesh by snapping the 2D rails and dragging each layer.

        Args:
            obj_list: Dragger-compatible object stack data. Each object is a
                list of z layers; every layer except the final sentinel must
                provide ``areas`` and ``element_size``.
            preserve_mesh2d: If true, copy the 2D node coordinates once before
                dragging. The default is zero-copy and mutates
                ``self.mesh2d.nodes`` in place as snap rules are applied.

        Returns:
            The populated :class:`Dragger` instance.
        """
        self._check_mesh_ready()
        nodes, elements = self._validate_mesh2d_arrays()

        mesh2d = self.mesh2d
        if preserve_mesh2d:
            mesh2d = Mesh2D(nodes=nodes.copy(), elements=elements)

        dragger_obj = Dragger()
        dragger_obj.set_2D(mesh2d)

        for obj in obj_list:
            if len(obj) < 2:
                continue

            self.reset_snap_state()
            dragger_obj._organize_empty()

            for layer_index, layer in enumerate(obj[:-1]):
                z_begin = obj[layer_index]["z"]
                z_end = obj[layer_index + 1]["z"]
                touched, touched_node_ids = self.apply_snap_rules_at_z(
                    z_begin,
                    nodes=dragger_obj.node_2D,
                    return_touched_node_ids=True,
                )

                if touched:
                    self._sync_dragger_current_layer_xy(
                        dragger_obj,
                        touched_node_ids,
                    )
                    dragger_obj._cal_volumns()

                dragger_obj._organize(layer["areas"], layer_index)
                dragger_obj._drag(layer["element_size"], z_begin, z_end)

        self.dragger = dragger_obj
        return dragger_obj

    def _check_mesh_ready(self):
        """Raise an error if the 2D mesh and rail lookup are unavailable."""
        if self.mesh2d is None or self.rail_node_index is None:
            raise RuntimeError(
                "Error: mesh_checkerboard or "
                "mesh_assignment is not performed"
            )

    def _sync_dragger_current_layer_xy(self, dragger_obj, node_ids):
        """Sync snapped 2D nodes into already-created top-layer 3D nodes."""
        node_ids = np.asarray(node_ids, dtype=np.intp)
        if node_ids.size == 0:
            return 0

        node3d_ids = dragger_obj.node_2D_to_3D[node_ids]
        valid_mask = node3d_ids >= 0
        if not np.any(valid_mask):
            return 0

        valid_node2d_ids = node_ids[valid_mask]
        valid_node3d_ids = node3d_ids[valid_mask]
        dragger_obj.nodes[valid_node3d_ids, :2] = dragger_obj.node_2D[
            valid_node2d_ids
        ]
        return int(len(valid_node3d_ids))
