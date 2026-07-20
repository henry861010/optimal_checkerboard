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
    Restore rules remain available as top-z lifecycle metadata.  Execution is
    order-independent: each call derives the complete active rule state at the
    requested exact z and restores nodes absent from that state.

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

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from typing import TYPE_CHECKING
import warnings

import numpy as np

from optimal_checkerboard.algorithms.classify_line import (
    _classify_line,
    _line_components,
)
from optimal_checkerboard.algorithms.feature_lines import (
    _extract_lines,
    _get_feature_lines,
    static_feature_span_endpoint_coordinates,
)
from optimal_checkerboard.algorithms.drag import Dragger
from optimal_checkerboard.algorithms.mesh_coverage import (
    validate_box_domain_partition,
    validate_mesh_coverage,
)
from optimal_checkerboard.algorithms.snap_plan_validation import (
    UnsafeSnapPlanError,
    validate_snap_plan,
)
from optimal_checkerboard.algorithms.snap_faces import snap_faces_to_shared_rails
from optimal_checkerboard.mesh import (
    build_structured_rail_node_index,
    generate_checkerboard_mesh,
    mesh_domain_from_obj,
    mesh_domain_pinned_coordinates,
    normalize_mesh_domain,
    structured_axes_for_box_domain,
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
    metadata: dict = None


class OptimalMesh25D:
    """Build and adjust a shared-rail checkerboard mesh for 2.5D extrusion.

    Public State:
        x_list: Shared x rail coordinates used by the checkerboard mesh.
        y_list: Shared y rail coordinates used by the checkerboard mesh.
        rails: Serialized rail metadata grouped by axis.
        snap_rules_by_z: Mapping from z value to snap rules.
        restore_rules_by_z: Top-z lifecycle metadata for feature rules.
        rail_node_index: Node ids on each rail, sorted by span coordinate.
        mesh2d: The generated or assigned 2D mesh.

    Typical users should call the public methods only.  The private helpers
    are intentionally small because future meshing variants may need to swap
    mesh generation, rail indexing, or snap execution independently.
    """

    def __init__(self):
        """Initialize an empty mesher state."""
        self.element_size = None
        self.merge_tolerance = None
        self.faces = None
        self.mesh_domain = None
        self.mandatory_coordinates = {"x": [], "y": []}
        self.pinned_coordinates = {"x": [], "y": []}
        self.rail_optimization_fallback = None
        self._rail_plan_is_exact = False
        self._pattern_integrity_signature = None

        self.group_lines_v = None
        self.group_lines_h = None
        self.x_list = None
        self.y_list = None
        self.mesh_x_list = None
        self.mesh_y_list = None

        self.rails = None
        self.snap_rules_by_z = None
        self.restore_rules_by_z = None
        self._rule_event_z_values = np.empty(0, dtype=np.float64)
        self.rail_reference_index = None
        self.rail_node_index = None
        self.node_axis_rail_ids = None

        self.mesh2d = None
        self.dragger = None
        self.x_pattern_nodes = None
        self.y_pattern_nodes = None

        # Pattern/mesh generations prevent a newly extracted pattern from
        # reusing structural node indices created for an older pattern.
        self._pattern_generation = 0
        self._mesh_pattern_generation = None

        # Snap execution is sparse.  Only coordinates changed by the previous
        # z state are restored before the next state is applied.
        self._current_changed_node_ids = {
            "x": np.empty(0, dtype=np.intp),
            "y": np.empty(0, dtype=np.intp),
        }
        # Exact expected coordinates for the managed state above.  Keeping the
        # sparse values lets integrity checks distinguish a legitimate snap
        # from an unmanaged edit to an already-snapped rail node without a
        # whole-mesh digest on every z event.
        self._current_changed_node_values = {
            "x": np.empty(0, dtype=np.float64),
            "y": np.empty(0, dtype=np.float64),
        }
        self._snap_state_nodes = None
        self._update_stamps = None
        self._update_targets = None
        self._update_generation = 0

        # Internal checkerboard meshes use arithmetic node-to-element lookup;
        # custom meshes fall back to a CSR adjacency built once at assignment.
        self._structured_grid_shape = None
        self._structured_axis_rail_ids = None
        self._node_element_offsets = None
        self._node_element_ids = None
        self._mesh_area_tolerance = None
        self._mesh_integrity_digest = None
        self._mesh_nodes_layout = None
        self._mesh_elements_layout = None
        self._structural_index_layout = None
        self._structural_index_digest = None

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
        element_size = float(element_size)
        ratio = float(ratio)
        if not np.isfinite(element_size) or element_size <= 0:
            raise ValueError("element_size must be a positive finite number")
        if not np.isfinite(ratio) or ratio < 0:
            raise ValueError("ratio must be a non-negative finite number")

        pattern_faces = deepcopy(faces)
        for face in pattern_faces:
            if face.get("type") == "BOX":
                x1, y1, x2, y2 = [float(value) for value in face["dim"]]
                face["dim"] = [
                    min(x1, x2),
                    min(y1, y2),
                    max(x1, x2),
                    max(y1, y2),
                ]
        normalized_domain = normalize_mesh_domain(mesh_domain)
        mandatory_coordinates = static_feature_span_endpoint_coordinates(
            pattern_faces,
        )
        domain_pins = mesh_domain_pinned_coordinates(normalized_domain)
        pinned_coordinates = {
            axis: sorted(
                set(mandatory_coordinates[axis]) | set(domain_pins[axis])
            )
            for axis in ("x", "y")
        }
        merge_tolerance = ratio * element_size
        if not np.isfinite(merge_tolerance):
            raise ValueError(
                "ratio * element_size must produce a finite merge tolerance"
            )
        feature_data = _get_feature_lines(
            pattern_faces,
            merge_tolerance,
            return_details=True,
            pinned_coords=pinned_coordinates,
        )
        fallback_reason = None
        try:
            self._validate_feature_data_snap_plan(
                feature_data,
                pinned_coordinates,
            )
        except UnsafeSnapPlanError as exc:
            fallback_reason = str(exc)
            feature_data = _get_feature_lines(
                pattern_faces,
                0.0,
                return_details=True,
                pinned_coords=pinned_coordinates,
            )
            self._validate_feature_data_snap_plan(
                feature_data,
                pinned_coordinates,
            )

        # Commit only after every geometry and rail validation succeeds.  A
        # malformed replacement pattern must not leave a half-updated mesher.
        self._invalidate_mesh_state()
        self._pattern_generation += 1
        self.element_size = element_size
        self.merge_tolerance = merge_tolerance
        self.faces = pattern_faces
        self.mesh_domain = normalized_domain
        self.mandatory_coordinates = mandatory_coordinates
        self.pinned_coordinates = pinned_coordinates
        self.rail_optimization_fallback = fallback_reason
        self._rail_plan_is_exact = fallback_reason is not None or merge_tolerance == 0.0
        self._install_feature_data(feature_data)
        self._clear_restore_state()

        return self.group_lines_v, self.group_lines_h, self.x_list, self.y_list

    def _install_feature_data(self, feature_data):
        """Install one already-validated rail plan and its event index."""
        (
            self.group_lines_v,
            self.group_lines_h,
            self.x_list,
            self.y_list,
            self.snap_rules_by_z,
            self.restore_rules_by_z,
            self.rails,
        ) = feature_data
        self._rule_event_z_values = np.asarray(
            sorted(
                {
                    float(value)
                    for rules in self.snap_rules_by_z.values()
                    for rule in rules
                    for value in (rule["z_bottom"], rule["z_top"])
                }
            ),
            dtype=np.float64,
        )
        self._pattern_integrity_signature = (
            self._current_pattern_integrity_signature()
        )

    def _validate_feature_data_snap_plan(
        self,
        feature_data,
        pinned_coordinates,
    ):
        """Preflight a rail lattice before any 2D mesh allocation."""
        axis_coordinates = {
            "x": sorted(set(feature_data[2]) | set(pinned_coordinates["x"])),
            "y": sorted(set(feature_data[3]) | set(pinned_coordinates["y"])),
        }
        if any(len(values) < 2 for values in axis_coordinates.values()):
            return None
        return validate_snap_plan(
            axis_coordinates,
            feature_data[6],
            feature_data[4],
        )

    def _fallback_to_exact_rails(self, reason):
        """Replace an unsafe optimized plan with exact-coordinate rails."""
        if self._rail_plan_is_exact:
            raise RuntimeError(
                "exact-coordinate rail plan failed safety preflight"
            ) from reason
        feature_data = _get_feature_lines(
            self.faces,
            0.0,
            return_details=True,
            pinned_coords=self.pinned_coordinates,
        )
        self._validate_feature_data_snap_plan(
            feature_data,
            self.pinned_coordinates,
        )
        self._invalidate_mesh_state()
        self._pattern_generation += 1
        self._install_feature_data(feature_data)
        self.rail_optimization_fallback = str(reason)
        self._rail_plan_is_exact = True

    def _invalidate_mesh_state(self):
        """Discard mesh state whose node indices belong to an older pattern."""
        self.mesh2d = None
        self.mesh_x_list = None
        self.mesh_y_list = None
        self.rail_reference_index = None
        self.rail_node_index = None
        self.node_axis_rail_ids = None
        self.dragger = None
        self._mesh_pattern_generation = None
        self._structured_grid_shape = None
        self._structured_axis_rail_ids = None
        self._node_element_offsets = None
        self._node_element_ids = None
        self._mesh_area_tolerance = None
        self._mesh_integrity_digest = None
        self._mesh_nodes_layout = None
        self._mesh_elements_layout = None
        self._structural_index_layout = None
        self._structural_index_digest = None
        self._update_stamps = None
        self._update_targets = None
        self._update_generation = 0
        self._current_changed_node_ids = {
            "x": np.empty(0, dtype=np.intp),
            "y": np.empty(0, dtype=np.intp),
        }
        self._current_changed_node_values = {
            "x": np.empty(0, dtype=np.float64),
            "y": np.empty(0, dtype=np.float64),
        }
        self._snap_state_nodes = None

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

        if self.mesh_domain["type"] == "BOX":
            x_nodes, y_nodes, _, _ = structured_axes_for_box_domain(
                self.mesh_domain,
                self.x_list,
                self.y_list,
                self.element_size,
                mandatory_coordinates=self.pinned_coordinates,
            )
            try:
                validate_snap_plan(
                    {"x": x_nodes, "y": y_nodes},
                    self.rails,
                    self.snap_rules_by_z,
                )
            except UnsafeSnapPlanError as exc:
                self._fallback_to_exact_rails(exc)
                return self._mesh_checkerboard_for_domain(
                    model_type=model_type,
                    center_x=center_x,
                    center_y=center_y,
                    required_domain_type=required_domain_type,
                )

        (
            nodes,
            elements,
            self.mesh_x_list,
            self.mesh_y_list,
            mesh_metadata,
        ) = generate_checkerboard_mesh(
            self.mesh_domain,
            self.x_list,
            self.y_list,
            self.element_size,
            mandatory_coordinates=self.pinned_coordinates,
            return_metadata=True,
        )

        self.mesh2d = Mesh2D(
            nodes=nodes,
            elements=elements,
            metadata=mesh_metadata,
        )
        try:
            self._validate_mesh2d_arrays()
            self._build_structured_rail_node_index(mesh_metadata)
            self._configure_mesh_execution_state(
                structured_grid_shape=mesh_metadata["grid_shape"],
            )
            self._validate_required_rail_coverage()
            self.reset_snap_state()
            self._validate_all_quad_orientations()
            self._record_mesh_integrity()
        except Exception:
            self._invalidate_mesh_state()
            raise
        self.dragger = None
        return self.mesh2d

    def mesh_assignment(self, mesh2d, structured_metadata=None):
        """Assign a user-provided 2D mesh and build rail lookup tables.

        Args:
            mesh2d: Object with ``nodes`` and ``elements`` attributes.  Nodes
                must exactly represent every shared rail and feature span.
            structured_metadata: Optional trusted-by-verification metadata for
                a complete row-major BOX grid.  It must contain ``kind`` equal
                to ``"STRUCTURED_BOX"``, ``grid_shape``, ``x_nodes``, and
                ``y_nodes``.  When omitted, the same metadata is read from
                ``mesh2d.metadata`` if present.  The arrays and connectivity
                are verified in bounded chunks before the compact structured
                execution path is enabled.

        Returns:
            The assigned mesh object.
        """
        self._check_pattern_ready()
        if (
            self.mesh_domain is not None
            and self.mesh_domain["type"] != "BOX"
        ):
            raise NotImplementedError(
                "verified custom-mesh domain partition is currently "
                "implemented only for BOX footprints"
            )

        old_state = self._capture_mesh_state()
        original_nodes = mesh2d.nodes
        original_elements = mesh2d.elements
        try:
            self.mesh2d = mesh2d
            self.mesh_x_list = np.asarray(self.x_list, dtype=np.float64)
            self.mesh_y_list = np.asarray(self.y_list, dtype=np.float64)
            self._validate_mesh2d_arrays()
            metadata = self._resolve_structured_assignment_metadata(
                mesh2d,
                structured_metadata,
            )
            if metadata is None:
                if (
                    self.mesh_domain is not None
                    and self.mesh_domain["type"] == "BOX"
                ):
                    validate_box_domain_partition(
                        self.mesh2d.nodes,
                        self.mesh2d.elements,
                        self.mesh_domain["bbox"],
                    )
                validate_mesh_coverage(
                    self.mesh2d.nodes,
                    self.mesh2d.elements,
                    self.rails,
                    self.snap_rules_by_z,
                    mandatory_coordinates=self.pinned_coordinates,
                    eps=0.0,
                )
                self._build_rail_node_index()
                self._configure_mesh_execution_state(
                    structured_grid_shape=None,
                )
            else:
                self._build_structured_rail_node_index(metadata)
                self._configure_mesh_execution_state(
                    structured_grid_shape=metadata["grid_shape"],
                )
            self._validate_required_rail_coverage()
            self.reset_snap_state()
            self._validate_all_quad_orientations()
            self._preflight_assigned_snap_states()
            self._record_mesh_integrity()
        except Exception:
            mesh2d.nodes = original_nodes
            mesh2d.elements = original_elements
            self._restore_mesh_state(old_state)
            raise
        self.dragger = None
        return self.mesh2d

    def _resolve_structured_assignment_metadata(
        self,
        mesh2d,
        structured_metadata,
    ):
        """Return verified compact metadata or ``None`` for a general mesh."""
        if structured_metadata is None:
            structured_metadata = getattr(mesh2d, "metadata", None)
        if structured_metadata is None:
            return None
        if not isinstance(structured_metadata, dict):
            raise ValueError("structured mesh metadata must be a dictionary")
        if structured_metadata.get("kind") != "STRUCTURED_BOX":
            raise ValueError(
                "structured mesh metadata kind must be STRUCTURED_BOX"
            )
        return self._validate_structured_assignment_metadata(
            structured_metadata,
        )

    def _validate_structured_assignment_metadata(
        self,
        metadata,
        chunk_size=250_000,
    ):
        """Verify row-major BOX topology without allocating mesh-sized maps."""
        try:
            shape = tuple(metadata["grid_shape"])
            raw_x_nodes = metadata["x_nodes"]
            raw_y_nodes = metadata["y_nodes"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "STRUCTURED_BOX metadata requires grid_shape, x_nodes, "
                "and y_nodes"
            ) from exc
        if (
            len(shape) != 2
            or any(isinstance(value, (bool, np.bool_)) for value in shape)
            or any(
                not isinstance(value, (int, np.integer))
                for value in shape
            )
        ):
            raise ValueError(
                "structured mesh grid_shape must contain two integers"
            )

        x_nodes = self._validated_structured_axis_metadata(
            raw_x_nodes,
            "x_nodes",
        )
        y_nodes = self._validated_structured_axis_metadata(
            raw_y_nodes,
            "y_nodes",
        )
        ny, nx = (int(shape[0]), int(shape[1]))
        if (ny, nx) != (len(y_nodes), len(x_nodes)):
            raise ValueError(
                "structured mesh grid_shape disagrees with x_nodes/y_nodes"
            )

        nodes = self.mesh2d.nodes
        elements = self.mesh2d.elements
        expected_node_count = nx * ny
        expected_element_count = (nx - 1) * (ny - 1)
        if (
            len(nodes) != expected_node_count
            or len(elements) != expected_element_count
        ):
            raise ValueError(
                "structured mesh metadata disagrees with mesh array sizes"
            )

        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            raise ValueError("structured metadata chunk_size must be positive")
        for start in range(0, expected_node_count, chunk_size):
            stop = min(start + chunk_size, expected_node_count)
            node_ids = np.arange(start, stop, dtype=np.intp)
            rows, cols = np.divmod(node_ids, nx)
            if (
                np.any(nodes[start:stop, 0] != x_nodes[cols])
                or np.any(nodes[start:stop, 1] != y_nodes[rows])
            ):
                raise ValueError(
                    "structured mesh nodes are not the declared exact "
                    "row-major x/y grid"
                )

        row_width = nx - 1
        for start in range(0, expected_element_count, chunk_size):
            stop = min(start + chunk_size, expected_element_count)
            element_ids = np.arange(start, stop, dtype=np.intp)
            rows, cols = np.divmod(element_ids, row_width)
            lower_left = rows * nx + cols
            chunk = elements[start:stop]
            if (
                np.any(chunk[:, 0] != lower_left)
                or np.any(chunk[:, 1] != lower_left + 1)
                or np.any(chunk[:, 2] != lower_left + nx + 1)
                or np.any(chunk[:, 3] != lower_left + nx)
            ):
                raise ValueError(
                    "structured mesh elements are not the declared exact "
                    "row-major quadrilateral topology"
                )

        if self.mesh_domain is not None and self.mesh_domain["type"] == "BOX":
            xmin, ymin, xmax, ymax = self.mesh_domain["bbox"]
            if (
                x_nodes[0] != xmin
                or x_nodes[-1] != xmax
                or y_nodes[0] != ymin
                or y_nodes[-1] != ymax
            ):
                raise ValueError(
                    "structured mesh axes do not exactly match the BOX domain"
                )

        self._validate_structured_axis_coverage(x_nodes, y_nodes)
        return {
            "kind": "STRUCTURED_BOX",
            "grid_shape": (ny, nx),
            # Own compact copies so later caller metadata mutation cannot
            # change rail spans behind the mesher's structural index.
            "x_nodes": x_nodes,
            "y_nodes": y_nodes,
        }

    @staticmethod
    def _validated_structured_axis_metadata(values, name):
        """Return one owned finite, strictly increasing float64 axis."""
        try:
            values = np.array(values, dtype=np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"structured mesh {name} must contain numeric coordinates"
            ) from exc
        if values.ndim != 1 or len(values) < 2:
            raise ValueError(
                f"structured mesh {name} must contain at least two values"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(
                f"structured mesh {name} coordinates must be finite"
            )
        intervals = np.diff(values)
        if np.any(~np.isfinite(intervals)):
            raise ValueError(
                f"structured mesh {name} intervals must be finite"
            )
        if np.any(intervals <= 0.0):
            raise ValueError(
                f"structured mesh {name} must be strictly increasing"
            )
        return values

    def _validate_structured_axis_coverage(self, x_nodes, y_nodes):
        """Prove rule and mandatory coverage from a verified full grid."""
        axis_nodes = {"x": x_nodes, "y": y_nodes}
        for axis in ("x", "y"):
            for rail in self.rails[axis]:
                self._require_exact_axis_coordinate(
                    axis_nodes[axis],
                    rail["coord"],
                    f"{axis}-axis rail",
                )
            for coordinate in self.pinned_coordinates[axis]:
                self._require_exact_axis_coordinate(
                    axis_nodes[axis],
                    coordinate,
                    f"mandatory {axis}-axis station",
                )

        for rules in self.snap_rules_by_z.values():
            for rule in rules:
                span_axis = "y" if rule["axis"] == "x" else "x"
                self._require_exact_axis_coordinate(
                    axis_nodes[span_axis],
                    rule["span_min"],
                    f"feature {rule['feature_id']} span endpoint",
                )
                self._require_exact_axis_coordinate(
                    axis_nodes[span_axis],
                    rule["span_max"],
                    f"feature {rule['feature_id']} span endpoint",
                )

    @staticmethod
    def _require_exact_axis_coordinate(axis_nodes, coordinate, label):
        """Require one float coordinate in a sorted structured axis."""
        coordinate = float(coordinate)
        index = int(np.searchsorted(axis_nodes, coordinate, side="left"))
        if index >= len(axis_nodes) or float(axis_nodes[index]) != coordinate:
            raise ValueError(f"structured mesh is missing exact {label}")

    def _preflight_assigned_snap_states(self):
        """Exercise every distinct custom-mesh snap state before commit."""
        boundaries = self._rule_event_z_values
        if not len(boundaries):
            return
        state_values = []
        for index, boundary in enumerate(boundaries):
            state_values.append(float(boundary))
            if index + 1 < len(boundaries):
                next_boundary = float(boundaries[index + 1])
                # Averaging the two halves stays finite even when both valid
                # event values are close to the float64 limit.
                state_values.append(
                    0.5 * float(boundary) + 0.5 * next_boundary
                )

        seen_signatures = set()
        try:
            for z_value in state_values:
                active_rules = self._active_snap_rules_at_z(z_value)
                signature = tuple(
                    self._snap_rule_geometry_key(rule)
                    for rule in active_rules
                )
                if signature in seen_signatures:
                    continue
                self._apply_snap_rule_batch(
                    self.mesh2d.nodes,
                    active_rules,
                    eps=0.0,
                )
                seen_signatures.add(signature)
        finally:
            self.reset_snap_state(nodes=self.mesh2d.nodes, sparse=True)

    def _capture_mesh_state(self):
        """Return the mesh-owned state needed for transactional assignment."""
        names = (
            "mesh2d",
            "mesh_x_list",
            "mesh_y_list",
            "rail_reference_index",
            "rail_node_index",
            "node_axis_rail_ids",
            "_mesh_pattern_generation",
            "_structured_grid_shape",
            "_structured_axis_rail_ids",
            "_node_element_offsets",
            "_node_element_ids",
            "_mesh_area_tolerance",
            "_mesh_integrity_digest",
            "_mesh_nodes_layout",
            "_mesh_elements_layout",
            "_structural_index_layout",
            "_structural_index_digest",
            "_update_stamps",
            "_update_targets",
            "_update_generation",
            "_current_changed_node_ids",
            "_current_changed_node_values",
            "_snap_state_nodes",
            "dragger",
        )
        return {name: getattr(self, name) for name in names}

    def _restore_mesh_state(self, state):
        """Restore a mesh state captured before a failed assignment."""
        for name, value in state.items():
            setattr(self, name, value)

    def get_snap_rules(self, z=None, eps=0.0):
        """Return all snap rules or the rules matching a specific z value.

        Args:
            z: Optional z coordinate.  If omitted, returns the full mapping.
            eps: Matching tolerance for z lookup.

        Returns:
            Either ``snap_rules_by_z`` or a list of rules for one z event.
        """
        if self.snap_rules_by_z is not None:
            self._assert_pattern_integrity()
        return deepcopy(
            self._get_rules_from_z_map(self.snap_rules_by_z, z, eps=eps)
        )

    def get_restore_rules(self, z=None, eps=0.0):
        """Return feature lifecycle metadata indexed by exact top z.

        Runtime snap execution does not consume this mapping; it derives the
        full active state from ``snap_rules_by_z`` on every call.
        """
        if self.restore_rules_by_z is not None:
            self._assert_pattern_integrity()
        return deepcopy(
            self._get_rules_from_z_map(
                self.restore_rules_by_z,
                z,
                eps=eps,
            )
        )

    def get_snap_faces(self, eps=0.0):
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

    def reset_snap_state(self, nodes=None, sparse=False):
        """Restore shared rails and clear managed state for a fresh traversal.

        Args:
            nodes: Optional mesh node array to restore. If omitted, restores
                ``self.mesh2d.nodes`` when a mesh has been indexed.
            sparse: Restore only nodes changed by the most recent managed snap
                state.  This keeps layer/object traversal proportional to the
                affected rail spans instead of the full mesh size.  The public
                default remains a complete indexed-rail reset so callers that
                modified node coordinates directly keep the historical
                behavior.

        Returns:
            Number of unique structural rail nodes restored to their baseline
            shared-rail coordinates.
        """
        self._clear_restore_state()

        if self.rail_node_index is None:
            return 0
        if self._mesh_integrity_digest is not None:
            self._assert_pattern_integrity()
            self._assert_authoritative_mesh_layout()
        if nodes is None:
            if self.mesh2d is None:
                return 0
            nodes = self.mesh2d.nodes
        nodes = np.asarray(nodes)
        self._bind_snap_state_nodes(nodes)
        if sparse:
            return self._restore_current_snap_state(nodes)
        restored = self._restore_shared_rail_baseline(nodes)
        self._current_changed_node_ids = {
            "x": np.empty(0, dtype=np.intp),
            "y": np.empty(0, dtype=np.intp),
        }
        self._current_changed_node_values = {
            "x": np.empty(0, dtype=np.float64),
            "y": np.empty(0, dtype=np.float64),
        }
        return restored

    def _clear_restore_state(self):
        """Compatibility no-op for the retired traversal-order buffer."""

    def _get_rules_from_z_map(self, rules_by_z, z=None, eps=0.0):
        """Return all z-indexed rules or one tolerant z bucket."""
        if rules_by_z is None:
            return {} if z is None else []

        if z is None:
            return rules_by_z

        _, rules = self._get_rules_and_z_from_z_map(rules_by_z, z, eps=eps)
        return rules

    def _get_rules_and_z_from_z_map(self, rules_by_z, z, eps=0.0):
        """Return the matched z key and rule bucket from a z-indexed map."""
        if rules_by_z is None:
            return None, []

        z = float(z)
        eps = float(eps)
        if not np.isfinite(z):
            raise ValueError("z must be finite")
        if not np.isfinite(eps) or eps < 0.0:
            raise ValueError("eps must be finite and non-negative")
        if z in rules_by_z:
            return z, rules_by_z[z]

        candidates = [
            (abs(float(z_key) - z), float(z_key), z_key, rules)
            for z_key, rules in rules_by_z.items()
            if abs(float(z_key) - z) <= eps
        ]
        if candidates:
            candidates.sort(key=lambda candidate: (candidate[0], candidate[1]))
            if (
                len(candidates) > 1
                and candidates[0][0] == candidates[1][0]
            ):
                raise ValueError(
                    "z lookup is ambiguous between two distinct rule events"
                )
            _, _, z_key, rules = candidates[0]
            return z_key, rules
        return None, []

    def _canonical_rule_z(self, z):
        """Return an unrounded z value retained for API compatibility."""
        return float(z)

    def _resolve_rule_state_z(self, z, eps):
        """Resolve representation noise without merging distinct z events."""
        z = float(z)
        eps = float(eps)
        if not np.isfinite(z):
            raise ValueError("z must be finite")
        if not np.isfinite(eps) or eps < 0.0:
            raise ValueError("eps must be finite and non-negative")
        events = self._rule_event_z_values
        if not len(events) or eps == 0.0:
            return z

        insertion = int(np.searchsorted(events, z, side="left"))
        if insertion < len(events) and float(events[insertion]) == z:
            return z
        candidates = []
        if insertion < len(events):
            candidates.append(float(events[insertion]))
        if insertion:
            candidates.append(float(events[insertion - 1]))
        candidates = [
            candidate for candidate in candidates if abs(candidate - z) <= eps
        ]
        if not candidates:
            return z
        candidates.sort(key=lambda candidate: (abs(candidate - z), candidate))
        if (
            len(candidates) > 1
            and abs(candidates[0] - z) == abs(candidates[1] - z)
        ):
            raise ValueError(
                "z is equally close to two distinct rule events; pass an "
                "exact event value or a smaller eps"
            )
        return candidates[0]

    def apply_snap_rules_at_z(
        self,
        z,
        nodes=None,
        eps=0.0,
        return_touched_node_ids=False,
    ):
        """Move rail nodes to true pattern coordinates.

        Args:
            z: Pattern z value whose complete active snap state is applied.
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
        self._assert_pattern_integrity()
        if (
            self.rail_node_index is None
            or self._mesh_pattern_generation != self._pattern_generation
        ):
            raise RuntimeError(
                "Error: mesh_checkerboard or "
                "mesh_assignment is not performed"
            )

        self._assert_authoritative_mesh_layout()
        if nodes is None:
            nodes = self.mesh2d.nodes

        nodes = np.asarray(nodes)
        if nodes.ndim != 2 or nodes.shape[1] < 2:
            raise ValueError("nodes must have shape (n, 2+)")
        if not np.issubdtype(nodes.dtype, np.floating):
            raise ValueError("nodes must contain floating point coordinates")
        self._bind_snap_state_nodes(nodes)

        z = self._resolve_rule_state_z(z, eps)
        active_rules = self._active_snap_rules_at_z(z)
        return self._apply_snap_rule_batch(
            nodes,
            active_rules,
            eps=0.0,
            return_touched_node_ids=return_touched_node_ids,
        )

    def _active_snap_rules_at_z(self, z):
        """Return snap rules active at z without depending on traversal order."""
        if self.snap_rules_by_z is None:
            return []

        z_value = float(z)
        active_rules = []
        seen_geometry = set()
        for rules in self.snap_rules_by_z.values():
            for rule in rules:
                if not (
                    float(rule["z_bottom"])
                    <= z_value
                    <= float(rule["z_top"])
                ):
                    continue
                geometry_key = self._snap_rule_geometry_key(rule)
                if geometry_key in seen_geometry:
                    continue
                seen_geometry.add(geometry_key)
                active_rules.append(rule)
        return active_rules

    def _snap_rule_geometry_key(self, rule):
        """Return lifecycle-independent geometry for runtime deduplication."""
        return (
            rule["axis"],
            rule["rail_id"],
            self._rule_rail_coord(rule),
            float(rule["target_coord"]),
            float(rule["span_min"]),
            float(rule["span_max"]),
        )

    def _apply_snap_rule_batch(
        self,
        nodes,
        rules,
        eps=0.0,
        return_touched_node_ids=False,
    ):
        """Apply the complete z state as a sparse, conflict-safe transaction."""
        self._ensure_update_buffers(len(nodes))
        rules_by_axis = self._rules_by_axis_and_rail(rules)
        moving_rules = [
            rule
            for rule in rules
            if float(rule["target_coord"]) != self._rule_rail_coord(rule)
        ]
        update_ids, update_values = self._collect_rule_updates(
            moving_rules,
            rules_by_axis,
            eps=eps,
        )

        old_changed = self._current_changed_node_ids
        affected_by_axis = {}
        old_values_by_axis = {}
        for axis, coord_axis in (("x", 0), ("y", 1)):
            chunks = [old_changed[axis], update_ids[axis]]
            nonempty = [chunk for chunk in chunks if len(chunk)]
            if nonempty:
                affected = np.unique(np.concatenate(nonempty)).astype(
                    np.intp,
                    copy=False,
                )
            else:
                affected = np.empty(0, dtype=np.intp)
            affected_by_axis[axis] = affected
            old_values_by_axis[axis] = nodes[affected, coord_axis].copy()

        # Restore only nodes changed by the previous state, then apply every
        # rule active at the requested z.  Rail lookup remains structural and
        # therefore independent of the nodes' current coordinates.
        for axis, coord_axis in (("x", 0), ("y", 1)):
            node_ids = old_changed[axis]
            if len(node_ids):
                nodes[node_ids, coord_axis] = self._baseline_coords_for_node_ids(
                    axis,
                    node_ids,
                )

            node_ids = update_ids[axis]
            if len(node_ids):
                nodes[node_ids, coord_axis] = update_values[axis]

        affected_nodes = [
            ids for ids in affected_by_axis.values() if len(ids)
        ]
        if affected_nodes:
            affected_node_ids = np.unique(np.concatenate(affected_nodes))
        else:
            affected_node_ids = np.empty(0, dtype=np.intp)

        try:
            self._validate_active_rule_geometry(
                nodes,
                moving_rules,
                rules_by_axis,
                eps=eps,
            )
            self._validate_local_quad_orientations(nodes, affected_node_ids)
        except Exception:
            for axis, coord_axis in (("x", 0), ("y", 1)):
                nodes[affected_by_axis[axis], coord_axis] = old_values_by_axis[
                    axis
                ]
            raise

        new_changed = {}
        actually_changed_chunks = []
        for axis, coord_axis in (("x", 0), ("y", 1)):
            affected = affected_by_axis[axis]
            if len(affected):
                baseline = self._baseline_coords_for_node_ids(axis, affected)
                changed_mask = nodes[affected, coord_axis] != baseline
                new_changed[axis] = affected[changed_mask]

                entry_changed = (
                    nodes[affected, coord_axis]
                    != old_values_by_axis[axis]
                )
                if np.any(entry_changed):
                    actually_changed_chunks.append(affected[entry_changed])
            else:
                new_changed[axis] = np.empty(0, dtype=np.intp)

        self._current_changed_node_ids = new_changed
        self._current_changed_node_values = {
            axis: nodes[node_ids, coord_axis].copy()
            for axis, coord_axis in (("x", 0), ("y", 1))
            for node_ids in (new_changed[axis],)
        }
        if actually_changed_chunks:
            touched_node_ids = np.unique(
                np.concatenate(actually_changed_chunks)
            ).astype(np.intp, copy=False)
        else:
            touched_node_ids = np.empty(0, dtype=np.intp)
        touched = int(len(touched_node_ids))

        if return_touched_node_ids:
            return touched, touched_node_ids
        return touched

    def _ensure_update_buffers(self, node_count):
        """Allocate reusable stamp/target buffers once per assigned 2D mesh."""
        if (
            self._update_stamps is not None
            and self._update_stamps.shape == (node_count,)
        ):
            return
        # One reusable axis buffer halves the persistent conflict-detection
        # footprint from 24 to 12 bytes per 2D node.  Sparse target values are
        # copied out before the same buffer is reused for the other axis.
        self._update_stamps = np.zeros(node_count, dtype=np.uint32)
        self._update_targets = np.empty(node_count, dtype=np.float64)
        self._update_generation = 0

    def _collect_rule_updates(self, rules, rules_by_axis, eps=0.0):
        """Collect unique sparse updates and reject conflicting targets."""
        chunks = {"x": [], "y": []}
        result = {}
        result_values = {}
        for axis in ("x", "y"):
            self._update_generation += 1
            if self._update_generation >= np.iinfo(np.uint32).max:
                self._update_stamps.fill(0)
                self._update_generation = 1
            generation = self._update_generation

            for rule in rules:
                if rule["axis"] != axis:
                    continue
                if float(rule["target_coord"]) == self._rule_rail_coord(rule):
                    continue
                node_ids = self._snap_rule_node_ids(
                    rule,
                    rules_by_axis,
                    eps=eps,
                )
                if len(node_ids) < 2:
                    raise ValueError(
                        "mesh2d cannot represent both endpoints of snap rule "
                        f"{rule['feature_id']} on the {axis}-axis"
                    )

                existing = self._update_stamps[node_ids] == generation
                target = float(rule["target_coord"])
                if np.any(existing):
                    previous = self._update_targets[node_ids[existing]]
                    if np.any(previous != target):
                        raise ValueError(
                            "Conflicting snap targets for the same structural "
                            f"node on the {axis}-axis at z={rule['z']}"
                        )

                new_ids = node_ids[~existing]
                if len(new_ids):
                    self._update_stamps[new_ids] = generation
                    self._update_targets[new_ids] = target
                    chunks[axis].append(new_ids)

            if chunks[axis]:
                result[axis] = np.unique(
                    np.concatenate(chunks[axis])
                ).astype(np.intp, copy=False)
                result_values[axis] = self._update_targets[
                    result[axis]
                ].copy()
            else:
                result[axis] = np.empty(0, dtype=np.intp)
                result_values[axis] = np.empty(0, dtype=np.float64)
        return result, result_values

    def _rule_rail_coord(self, rule):
        """Return a rule's structural coordinate, including legacy rules."""
        if "rail_coord" in rule:
            return float(rule["rail_coord"])
        return float(self.rails[rule["axis"]][rule["rail_id"]]["coord"])

    def _validate_active_rule_geometry(
        self,
        nodes,
        rules,
        rules_by_axis,
        eps=0.0,
    ):
        """Require every active feature span to retain both exact endpoints."""
        for rule in rules:
            node_ids = self._snap_rule_node_ids(
                rule,
                rules_by_axis,
                eps=eps,
            )
            span_axis = 1 if rule["axis"] == "x" else 0
            coord_axis = 0 if rule["axis"] == "x" else 1
            span_values = nodes[node_ids, span_axis]
            if not (
                np.any(span_values == float(rule["span_min"]))
                and np.any(span_values == float(rule["span_max"]))
                and np.all(
                    nodes[node_ids, coord_axis]
                    == float(rule["target_coord"])
                )
            ):
                raise ValueError(
                    "Snap state cannot represent the exact endpoints of "
                    f"feature {rule['feature_id']} on the {rule['axis']}-axis"
                )

    def _baseline_coords_for_node_ids(self, axis, node_ids):
        """Return structural rail coordinates for indexed node ids."""
        if self._structured_axis_rail_ids is not None:
            _, nx = self._structured_grid_shape
            axis_indices = node_ids % nx if axis == "x" else node_ids // nx
            rail_ids = self._structured_axis_rail_ids[axis][axis_indices]
        else:
            rail_ids = self.node_axis_rail_ids[axis][node_ids]
        if np.any(rail_ids < 0):
            raise ValueError(
                f"Snap update references nodes outside indexed {axis}-rails"
            )
        rail_coords = np.asarray(
            [rail["coord"] for rail in self.rails[axis]],
            dtype=np.float64,
        )
        return rail_coords[rail_ids]

    def _apply_snap_rule(self, nodes, rule, eps=0.0):
        """Apply one snap rule to a sorted rail-node span."""
        node_ids = self._span_node_ids(rule, eps=eps)

        coord_axis = 0 if rule["axis"] == "x" else 1
        nodes[node_ids, coord_axis] = rule["target_coord"]
        return int(len(node_ids))

    def _span_node_ids(self, rule, eps=0.0):
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

    def _snap_rule_node_ids(self, rule, rules_by_axis, eps=0.0):
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

    def _coupled_corner_node_ids(self, rule, rules_by_axis, eps=0.0):
        """Return corner nodes whose opposite rail snaps into this span."""
        axis = rule["axis"]
        opposite_axis = "y" if axis == "x" else "x"
        opposite_rule_lists = rules_by_axis[opposite_axis]
        if not opposite_rule_lists:
            return np.empty(0, dtype=np.intp)

        matching_node_ids = []
        rail_index = self.rail_node_index[axis][rule["rail_id"]]
        span_values = rail_index["span_values"]
        rail_node_ids = rail_index["node_ids"]
        for opposite_rules in opposite_rule_lists.values():
            for opposite_rule in opposite_rules:
                if not self._rule_targets_cross(rule, opposite_rule, eps=eps):
                    continue
                structural_cross_coord = self._rule_rail_coord(opposite_rule)
                lo = np.searchsorted(
                    span_values,
                    structural_cross_coord - eps,
                    side="left",
                )
                hi = np.searchsorted(
                    span_values,
                    structural_cross_coord + eps,
                    side="right",
                )
                if hi > lo:
                    matching_node_ids.append(rail_node_ids[lo:hi])

        if not matching_node_ids:
            return np.empty(0, dtype=np.intp)
        return np.unique(np.concatenate(matching_node_ids))

    def _rule_targets_cross(self, rule, opposite_rule, eps=0.0):
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
        self._assert_pattern_integrity()

    @classmethod
    def _freeze_integrity_value(cls, value):
        """Return a deterministic immutable representation of pattern data."""
        if isinstance(value, np.generic):
            return cls._freeze_integrity_value(value.item())
        if isinstance(value, np.ndarray):
            contiguous = np.ascontiguousarray(value)
            digest = hashlib.blake2b(
                memoryview(contiguous).cast("B"),
                digest_size=16,
            ).digest()
            return (
                "ndarray",
                value.dtype.str,
                tuple(value.shape),
                digest,
            )
        if isinstance(value, dict):
            items = [
                (
                    cls._freeze_integrity_value(key),
                    cls._freeze_integrity_value(item),
                )
                for key, item in value.items()
            ]
            items.sort(key=lambda pair: repr(pair[0]))
            return ("dict", tuple(items))
        if isinstance(value, (list, tuple)):
            return (
                type(value).__name__,
                tuple(cls._freeze_integrity_value(item) for item in value),
            )
        if isinstance(value, (set, frozenset)):
            frozen = [cls._freeze_integrity_value(item) for item in value]
            frozen.sort(key=repr)
            return (type(value).__name__, tuple(frozen))
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        raise TypeError(
            "pattern integrity contains unsupported value type "
            f"{type(value).__name__}"
        )

    def _current_pattern_integrity_signature(self):
        """Return a geometry-scale signature of every execution input."""
        return self._freeze_integrity_value(
            {
                "element_size": self.element_size,
                "merge_tolerance": self.merge_tolerance,
                "faces": self.faces,
                "mesh_domain": self.mesh_domain,
                "mandatory_coordinates": self.mandatory_coordinates,
                "pinned_coordinates": self.pinned_coordinates,
                "x_list": self.x_list,
                "y_list": self.y_list,
                "rails": self.rails,
                "snap_rules_by_z": self.snap_rules_by_z,
                "restore_rules_by_z": self.restore_rules_by_z,
                "rule_event_z_values": self._rule_event_z_values,
            }
        )

    def _assert_pattern_integrity(self):
        """Reject public pattern-plan mutation after validated extraction."""
        expected = self._pattern_integrity_signature
        if expected is None:
            raise RuntimeError(
                "pattern integrity state is unavailable; call set_pattern_obj"
            )
        if self._current_pattern_integrity_signature() != expected:
            raise RuntimeError(
                "pattern faces, rails, or snap rules changed after "
                "validation; call set_pattern_obj again"
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
        if nodes.dtype != np.float64:
            nodes = nodes.astype(np.float64)
        if not np.all(np.isfinite(nodes[:, :2])):
            raise ValueError("mesh2d.nodes must contain finite xy coordinates")
        if len(nodes) - 1 > np.iinfo(np.int32).max:
            raise OverflowError(
                "mesh2d node count exceeds int32 connectivity capacity"
            )

        if elements.ndim != 2 or elements.shape[1] != 4:
            raise ValueError("mesh2d.elements must have shape (m, 4)")
        if not np.issubdtype(elements.dtype, np.integer):
            raise ValueError("mesh2d.elements must contain integer node ids")
        if len(elements) > np.iinfo(np.int32).max:
            raise OverflowError(
                "mesh2d element count exceeds int32 index capacity"
            )

        if elements.size:
            min_node_id = int(elements.min())
            max_node_id = int(elements.max())
            if min_node_id < 0:
                raise ValueError(
                    "mesh2d.elements contain node ids outside mesh2d.nodes"
                )
            if max_node_id > np.iinfo(np.int32).max:
                raise OverflowError(
                    "mesh2d.elements exceed int32 connectivity capacity"
                )
            if max_node_id >= len(nodes):
                raise ValueError(
                    "mesh2d.elements contain node ids outside mesh2d.nodes"
                )
        if elements.dtype != np.int32:
            elements = elements.astype(np.int32)
        self.mesh2d.nodes = nodes
        self.mesh2d.elements = elements
        return nodes, elements

    @staticmethod
    def _array_layout_signature(array):
        """Return an O(1) identity/layout token for one public mesh array."""
        array = np.asarray(array)
        return (
            id(array),
            tuple(array.shape),
            array.dtype.str,
            tuple(array.strides),
            int(array.ctypes.data),
        )

    def _record_mesh_integrity(self):
        """Record one exact baseline digest and lock structural connectivity."""
        nodes, elements = self._validate_mesh2d_arrays()
        mesh_integrity_digest = self._calculate_mesh_integrity_digest(
            nodes,
            elements,
        )
        mesh_nodes_layout = self._array_layout_signature(nodes)
        mesh_elements_layout = self._array_layout_signature(elements)
        structural_index_layout = self._current_structural_index_layout()
        structural_index_digest = self._freeze_integrity_value(
            self._structural_index_data()
        )
        # Snap execution mutates coordinates through managed methods, but
        # connectivity must remain structural.  Callers that need to edit it
        # must assign a copied mesh again so all indices are rebuilt.
        # Compute every potentially failing signature before changing the
        # caller-owned connectivity flag.  Structural arrays are internal and
        # can be discarded transactionally if their lock unexpectedly fails.
        self._lock_structural_index_arrays()
        elements.setflags(write=False)
        self._mesh_integrity_digest = mesh_integrity_digest
        self._mesh_nodes_layout = mesh_nodes_layout
        self._mesh_elements_layout = mesh_elements_layout
        self._structural_index_layout = structural_index_layout
        self._structural_index_digest = structural_index_digest

    def _structural_index_data(self):
        """Return every array/scalar that maps snap nodes to mesh topology."""
        rail_data = {"x": [], "y": []}
        if self.rail_node_index is not None:
            for axis in ("x", "y"):
                for rail in self.rail_node_index[axis]:
                    rail_data[axis].append(
                        {
                            "coord": rail["coord"],
                            "node_ids": rail["node_ids"],
                            "span_values": rail["span_values"],
                        }
                    )
        return {
            "rail_node_index": rail_data,
            "node_axis_rail_ids": self.node_axis_rail_ids,
            "structured_grid_shape": self._structured_grid_shape,
            "structured_axis_rail_ids": self._structured_axis_rail_ids,
            "node_element_offsets": self._node_element_offsets,
            "node_element_ids": self._node_element_ids,
        }

    def _structural_index_arrays(self):
        """Yield structural arrays without materializing mesh-sized copies."""
        if self.rail_node_index is not None:
            for axis in ("x", "y"):
                for rail_id, rail in enumerate(self.rail_node_index[axis]):
                    yield (
                        f"rail_node_index.{axis}.{rail_id}.node_ids",
                        rail["node_ids"],
                    )
                    yield (
                        f"rail_node_index.{axis}.{rail_id}.span_values",
                        rail["span_values"],
                    )
        for name, mapping in (
            ("node_axis_rail_ids", self.node_axis_rail_ids),
            ("structured_axis_rail_ids", self._structured_axis_rail_ids),
        ):
            if mapping is not None:
                for axis in ("x", "y"):
                    yield f"{name}.{axis}", mapping[axis]
        for name, array in (
            ("node_element_offsets", self._node_element_offsets),
            ("node_element_ids", self._node_element_ids),
        ):
            if array is not None:
                yield name, array

    def _lock_structural_index_arrays(self):
        """Make ordinary in-place structural-index mutation fail immediately."""
        for _, array in self._structural_index_arrays():
            np.asarray(array).setflags(write=False)

    def _current_structural_index_layout(self):
        """Return O(number of rails) scalar and identity/layout tokens."""
        rail_coordinates = ()
        if self.rail_node_index is not None:
            rail_coordinates = tuple(
                (
                    axis,
                    tuple(
                        float(rail["coord"])
                        for rail in self.rail_node_index[axis]
                    ),
                )
                for axis in ("x", "y")
            )
        return (
            rail_coordinates,
            self._structured_grid_shape,
            tuple(
                (name, self._array_layout_signature(array))
                for name, array in self._structural_index_arrays()
            ),
        )

    @staticmethod
    def _calculate_mesh_integrity_digest(
        nodes,
        elements,
        chunk_size=250_000,
        node_coordinate_overrides=None,
    ):
        """Hash baseline xy and connectivity with bounded temporary memory."""
        hasher = hashlib.blake2b(digest_size=32)
        hasher.update(b"optimal-checkerboard-mesh-v1\0")
        hasher.update(nodes.dtype.str.encode("ascii"))
        hasher.update(np.asarray(nodes.shape, dtype=np.int64).tobytes())
        overrides = node_coordinate_overrides or {}
        for start in range(0, len(nodes), chunk_size):
            stop = min(start + chunk_size, len(nodes))
            chunk = np.ascontiguousarray(nodes[start:stop, :2])
            for coord_axis in (0, 1):
                axis_override = overrides.get(coord_axis)
                if axis_override is None:
                    continue
                node_ids, baseline_values = axis_override
                lo = int(np.searchsorted(node_ids, start, side="left"))
                hi = int(np.searchsorted(node_ids, stop, side="left"))
                if hi > lo:
                    chunk[node_ids[lo:hi] - start, coord_axis] = (
                        baseline_values[lo:hi]
                    )
            hasher.update(memoryview(chunk).cast("B"))

        hasher.update(elements.dtype.str.encode("ascii"))
        hasher.update(np.asarray(elements.shape, dtype=np.int64).tobytes())
        for start in range(0, len(elements), chunk_size):
            chunk = np.ascontiguousarray(elements[start : start + chunk_size, :4])
            hasher.update(memoryview(chunk).cast("B"))
        return hasher.digest()

    def _managed_baseline_coordinate_overrides(
        self,
        nodes,
        *,
        require_same_buffer,
    ):
        """Validate one managed snap state and return sparse baseline values.

        A baseline digest must normalize legitimate snap coordinates, but it
        must not overwrite an unmanaged edit before checking it.  Expected
        managed values are therefore compared exactly first; only the bounded
        digest chunks receive the returned baseline substitutions.
        """
        has_changes = any(
            len(node_ids)
            for node_ids in self._current_changed_node_ids.values()
        )
        if not has_changes:
            return {}
        if (
            self._snap_state_nodes is None
            or not self._same_xy_storage(self._snap_state_nodes, nodes)
        ):
            if require_same_buffer:
                raise RuntimeError(
                    "snap state belongs to a different node buffer; reset that "
                    "buffer before building"
                )
            return {}

        overrides = {}
        for axis, coord_axis in (("x", 0), ("y", 1)):
            node_ids = np.asarray(
                self._current_changed_node_ids.get(axis),
            )
            expected = np.asarray(
                self._current_changed_node_values.get(axis),
            )
            if (
                node_ids.ndim != 1
                or not np.issubdtype(node_ids.dtype, np.integer)
                or expected.shape != node_ids.shape
                or not np.issubdtype(expected.dtype, np.floating)
                or (
                    len(node_ids) > 1
                    and np.any(node_ids[1:] <= node_ids[:-1])
                )
                or (
                    len(node_ids)
                    and (
                        int(node_ids[0]) < 0
                        or int(node_ids[-1]) >= len(nodes)
                    )
                )
            ):
                raise RuntimeError("managed snap-state bookkeeping is invalid")
            node_ids = node_ids.astype(np.intp, copy=False)
            expected = expected.astype(np.float64, copy=False)
            if np.any(nodes[node_ids, coord_axis] != expected):
                raise RuntimeError(
                    "mesh2d nodes changed outside managed snap operations; "
                    "regenerate or reassign the 2D mesh"
                )
            if len(node_ids):
                overrides[coord_axis] = (
                    node_ids,
                    self._baseline_coords_for_node_ids(axis, node_ids),
                )
        return overrides

    @staticmethod
    def _apply_baseline_coordinate_overrides(nodes, overrides):
        """Apply already-validated sparse baseline substitutions in place."""
        for coord_axis, (node_ids, baseline_values) in overrides.items():
            nodes[node_ids, coord_axis] = baseline_values

    def _assert_authoritative_mesh_layout(self):
        """Reject replaced public arrays before using structural indices."""
        if (
            self._mesh_integrity_digest is None
            or self._mesh_nodes_layout is None
            or self._mesh_elements_layout is None
        ):
            raise RuntimeError(
                "mesh integrity state is unavailable; regenerate or reassign "
                "the 2D mesh"
            )
        if (
            self._array_layout_signature(self.mesh2d.nodes)
            != self._mesh_nodes_layout
            or self._array_layout_signature(self.mesh2d.elements)
            != self._mesh_elements_layout
        ):
            raise RuntimeError(
                "mesh2d arrays were replaced after indexing; call "
                "mesh_assignment again"
            )
        if (
            self._structural_index_layout is None
            or self._current_structural_index_layout()
            != self._structural_index_layout
        ):
            raise RuntimeError(
                "mesh structural indices were replaced after validation; "
                "regenerate or reassign the 2D mesh"
            )

    def _assert_mesh_integrity(
        self,
        baseline_nodes,
        elements,
        node_coordinate_overrides=None,
    ):
        """Reject any in-place public mesh mutation before 3D extrusion."""
        self._assert_authoritative_mesh_layout()
        actual = self._calculate_mesh_integrity_digest(
            baseline_nodes,
            elements,
            node_coordinate_overrides=node_coordinate_overrides,
        )
        if actual != self._mesh_integrity_digest:
            raise RuntimeError(
                "mesh2d nodes or elements changed after validation; call "
                "mesh_assignment again before build"
            )
        structural_digest = self._freeze_integrity_value(
            self._structural_index_data()
        )
        if structural_digest != self._structural_index_digest:
            raise RuntimeError(
                "mesh structural indices changed after validation; regenerate "
                "or reassign the 2D mesh"
            )

    def _configure_mesh_execution_state(
        self,
        structured_grid_shape=None,
        structured=None,
    ):
        """Configure sparse snap and local element validation indices."""
        nodes, elements = self._validate_mesh2d_arrays()
        # A successful assignment records a fresh integrity baseline after
        # every structural index has been built and validated.  Clear an old
        # mesh's record first so setup-time reset/validation cannot compare
        # the new indices with stale identity tokens.
        self._mesh_integrity_digest = None
        self._mesh_nodes_layout = None
        self._mesh_elements_layout = None
        self._structural_index_layout = None
        self._structural_index_digest = None
        if structured is not None:
            if structured_grid_shape is not None:
                raise ValueError(
                    "provide structured or structured_grid_shape, not both"
                )
            if structured:
                structured_grid_shape = self._infer_structured_grid_shape(
                    nodes,
                    elements,
                )
        self._mesh_pattern_generation = self._pattern_generation
        self._ensure_update_buffers(len(nodes))
        self._mesh_area_tolerance = 0.0
        self._current_changed_node_ids = {
            "x": np.empty(0, dtype=np.intp),
            "y": np.empty(0, dtype=np.intp),
        }
        self._current_changed_node_values = {
            "x": np.empty(0, dtype=np.float64),
            "y": np.empty(0, dtype=np.float64),
        }
        self._snap_state_nodes = nodes

        self._structured_grid_shape = None
        if structured_grid_shape is None:
            self._structured_axis_rail_ids = None
        self._node_element_offsets = None
        self._node_element_ids = None
        if structured_grid_shape is not None:
            ny, nx = [int(value) for value in structured_grid_shape]
            if (
                nx < 2
                or ny < 2
                or len(nodes) != nx * ny
                or len(elements) != (nx - 1) * (ny - 1)
            ):
                raise ValueError("invalid structured checkerboard metadata")
            self._structured_grid_shape = (ny, nx)
        else:
            self._build_node_element_csr(len(nodes), elements)

    def _bind_snap_state_nodes(self, nodes):
        """Bind sparse restore bookkeeping to one concrete node buffer."""
        nodes = np.asarray(nodes)
        if nodes.ndim != 2 or nodes.shape[1] < 2:
            raise ValueError("nodes must have shape (n, 2+)")
        if self.mesh2d is not None and len(nodes) != len(self.mesh2d.nodes):
            raise ValueError("nodes must contain the complete indexed 2D mesh")

        current = self._snap_state_nodes
        if current is None or self._same_xy_storage(current, nodes):
            if current is None:
                self._snap_state_nodes = nodes
            return
        if any(len(ids) for ids in self._current_changed_node_ids.values()):
            raise RuntimeError(
                "snap state belongs to a different node buffer; reset that "
                "buffer before switching nodes"
            )
        self._snap_state_nodes = nodes
        self._current_changed_node_ids = {
            "x": np.empty(0, dtype=np.intp),
            "y": np.empty(0, dtype=np.intp),
        }
        self._current_changed_node_values = {
            "x": np.empty(0, dtype=np.float64),
            "y": np.empty(0, dtype=np.float64),
        }

    def _same_xy_storage(self, left, right):
        """Return whether two full arrays expose the same indexed xy data."""
        if left is right:
            return True
        if left.shape[0] != right.shape[0]:
            return False
        return (
            np.shares_memory(left[:, :2], right[:, :2])
            and left[:, :2].ctypes.data == right[:, :2].ctypes.data
            and left[:, :2].strides == right[:, :2].strides
        )

    def _infer_structured_grid_shape(self, nodes, elements, eps=0.0):
        """Return (ny, nx) for the row-major internal checkerboard grid."""
        if len(nodes) < 4:
            raise ValueError("checkerboard mesh must contain at least four nodes")

        first_y = nodes[0, 1]
        if eps != 0.0:
            raise ValueError("structured topology inference requires eps=0")
        row_breaks = np.flatnonzero(nodes[:, 1] != first_y)
        nx = int(row_breaks[0]) if len(row_breaks) else len(nodes)
        if nx < 2 or len(nodes) % nx:
            raise ValueError("generated checkerboard nodes are not a regular grid")
        ny = len(nodes) // nx
        if ny < 2 or len(elements) != (nx - 1) * (ny - 1):
            raise ValueError("generated checkerboard connectivity is not structured")
        return ny, nx

    def _build_node_element_csr(self, node_count, elements):
        """Build one compact node-to-element adjacency for a custom mesh."""
        if not elements.size:
            self._node_element_offsets = np.zeros(node_count + 1, dtype=np.intp)
            self._node_element_ids = np.empty(0, dtype=np.intp)
            return

        flat_node_ids = elements.ravel().astype(np.intp, copy=False)
        counts = np.bincount(flat_node_ids, minlength=node_count)
        offsets = np.empty(node_count + 1, dtype=np.intp)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])

        order = np.argsort(flat_node_ids, kind="stable")
        element_dtype = np.int32 if len(elements) < 2**31 else np.int64
        flat_element_ids = np.repeat(
            np.arange(len(elements), dtype=element_dtype),
            elements.shape[1],
        )
        self._node_element_offsets = offsets
        self._node_element_ids = flat_element_ids[order]

    def _incident_element_ids(self, node_ids):
        """Return unique elements touching node_ids without a full mesh scan."""
        node_ids = np.unique(np.asarray(node_ids, dtype=np.intp))
        if not len(node_ids):
            return np.empty(0, dtype=np.intp)

        if self._structured_grid_shape is not None:
            ny, nx = self._structured_grid_shape
            rows = node_ids // nx
            cols = node_ids % nx
            chunks = []
            for row_offset, col_offset in (
                (-1, -1),
                (-1, 0),
                (0, -1),
                (0, 0),
            ):
                element_rows = rows + row_offset
                element_cols = cols + col_offset
                valid = (
                    (element_rows >= 0)
                    & (element_rows < ny - 1)
                    & (element_cols >= 0)
                    & (element_cols < nx - 1)
                )
                if np.any(valid):
                    chunks.append(
                        element_rows[valid] * (nx - 1) + element_cols[valid]
                    )
            if not chunks:
                return np.empty(0, dtype=np.intp)
            return np.unique(np.concatenate(chunks)).astype(
                np.intp,
                copy=False,
            )

        offsets = self._node_element_offsets
        lengths = offsets[node_ids + 1] - offsets[node_ids]
        total = int(lengths.sum())
        if total == 0:
            return np.empty(0, dtype=np.intp)
        group_starts = np.cumsum(lengths) - lengths
        positions = (
            np.repeat(offsets[node_ids], lengths)
            + np.arange(total, dtype=np.intp)
            - np.repeat(group_starts, lengths)
        )
        return np.unique(self._node_element_ids[positions]).astype(
            np.intp,
            copy=False,
        )

    def _quad_validity_metrics(self, nodes, element_ids):
        """Return stable signed areas and corner turns for selected quads."""
        element_ids = np.asarray(element_ids, dtype=np.intp)
        if not len(element_ids):
            return (
                np.empty(0, dtype=np.float64),
                np.empty((0, 4), dtype=np.float64),
            )
        points = nodes[self.mesh2d.elements[element_ids], :2]
        shifted = points - points[:, :1, :]
        edges = np.roll(shifted, -1, axis=1) - shifted
        corner_crosses = (
            edges[:, :, 0] * np.roll(edges[:, :, 1], -1, axis=1)
            - edges[:, :, 1] * np.roll(edges[:, :, 0], -1, axis=1)
        )
        area2 = (
            shifted[:, 1, 0] * shifted[:, 2, 1]
            - shifted[:, 1, 1] * shifted[:, 2, 0]
            + shifted[:, 2, 0] * shifted[:, 3, 1]
            - shifted[:, 2, 1] * shifted[:, 3, 0]
        )
        return 0.5 * area2, corner_crosses

    def _signed_quad_areas(self, nodes, element_ids):
        """Return cancellation-resistant signed areas for selected quads."""
        areas, _ = self._quad_validity_metrics(nodes, element_ids)
        return areas

    def _quad_area_tolerance(self, nodes):
        """Return a scale-aware strict-positive area tolerance."""
        return 0.0

    def _validate_all_quad_orientations(self, chunk_size=250_000):
        """Validate the assigned 2D mesh once using bounded temporaries."""
        nodes, elements = self._validate_mesh2d_arrays()
        tolerance = self._mesh_area_tolerance
        for start in range(0, len(elements), chunk_size):
            stop = min(start + chunk_size, len(elements))
            element_ids = np.arange(start, stop, dtype=np.intp)
            areas, corner_crosses = self._quad_validity_metrics(
                nodes,
                element_ids,
            )
            if (
                np.any(~np.isfinite(areas))
                or np.any(~np.isfinite(corner_crosses))
                or np.any(areas <= tolerance)
                or np.any(corner_crosses <= 0.0)
            ):
                raise ValueError(
                    "mesh2d contains degenerate, concave, self-intersecting, "
                    "or reversed quadrilateral elements"
                )

    def _validate_local_quad_orientations(
        self,
        nodes,
        node_ids,
        chunk_size=250_000,
    ):
        """Validate only elements incident to a sparse snap transaction."""
        element_ids = self._incident_element_ids(node_ids)
        if not len(element_ids):
            return
        tolerance = self._mesh_area_tolerance
        for start in range(0, len(element_ids), chunk_size):
            selected = element_ids[start : start + chunk_size]
            areas, corner_crosses = self._quad_validity_metrics(
                nodes,
                selected,
            )
            if (
                np.any(~np.isfinite(areas))
                or np.any(~np.isfinite(corner_crosses))
                or np.any(areas <= tolerance)
                or np.any(corner_crosses <= 0.0)
            ):
                raise ValueError(
                    "Snap would create a degenerate, concave, "
                    "self-intersecting, or reversed quadrilateral element"
                )

    def _build_rail_node_index(self, eps=0.0):
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

    def _build_structured_rail_node_index(self, metadata):
        """Install rail lookup using generated axis arrays, without sorting."""
        if metadata.get("kind") != "STRUCTURED_BOX":
            raise ValueError("unsupported structured checkerboard metadata")
        self.rail_node_index = build_structured_rail_node_index(
            metadata["x_nodes"],
            metadata["y_nodes"],
            self.rails,
            # These arrays were generated from the exact same rail values.
            # A nonzero search tolerance could alias two extremely close but
            # intentionally distinct pattern coordinates.
            eps=0.0,
        )
        x_nodes = metadata["x_nodes"]
        y_nodes = metadata["y_nodes"]
        self._structured_axis_rail_ids = {
            "x": np.full(len(x_nodes), -1, dtype=np.int32),
            "y": np.full(len(y_nodes), -1, dtype=np.int32),
        }
        for axis, axis_nodes in (("x", x_nodes), ("y", y_nodes)):
            for rail_id, rail in enumerate(self.rails[axis]):
                axis_index = int(np.searchsorted(axis_nodes, rail["coord"]))
                if (
                    axis_index >= len(axis_nodes)
                    or float(axis_nodes[axis_index]) != float(rail["coord"])
                ):
                    raise ValueError(
                        f"structured mesh is missing {axis}-axis rail"
                    )
                existing = self._structured_axis_rail_ids[axis][axis_index]
                if existing >= 0 and existing != rail_id:
                    raise ValueError(
                        "one structured axis station cannot represent two rails"
                    )
                self._structured_axis_rail_ids[axis][axis_index] = rail_id
        self.node_axis_rail_ids = None
        self.rail_reference_index = self.rail_node_index

    def _build_node_axis_rail_ids(self, node_count):
        """Return each node's structural rail id for both axes."""
        node_axis_rail_ids = {
            "x": np.full(node_count, -1, dtype=np.intp),
            "y": np.full(node_count, -1, dtype=np.intp),
        }

        for axis, rail_indices in self.rail_node_index.items():
            for rail_id, rail_index in enumerate(rail_indices):
                existing = node_axis_rail_ids[axis][rail_index["node_ids"]]
                if np.any((existing >= 0) & (existing != rail_id)):
                    raise ValueError(
                        "one mesh node cannot represent two distinct "
                        f"{axis}-axis structural rails"
                    )
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
                    "coord": float(rail["coord"]),
                    "node_ids": active_node_ids[node_refs],
                    "span_values": span_values[node_refs],
                }
            )

        return rail_indices

    def _validate_required_rail_coverage(self, eps=0.0):
        """Reject assigned meshes that cannot execute every snap rule."""
        for axis, rail_indices in self.rail_node_index.items():
            for rail_id, rail_index in enumerate(rail_indices):
                if len(rail_index["node_ids"]):
                    continue
                raise ValueError(
                    "mesh2d is missing active nodes for required "
                    f"{axis}-axis shared rail {rail_id} at "
                    f"coordinate {rail_index['coord']}"
                )

        all_rules = [
            rule
            for rules in self.snap_rules_by_z.values()
            for rule in rules
        ]
        rules_by_axis = self._rules_by_axis_and_rail(all_rules)
        for z, rules in self.snap_rules_by_z.items():
            for rule in rules:
                node_ids = self._snap_rule_node_ids(
                    rule,
                    rules_by_axis,
                    eps=eps,
                )
                if len(node_ids) >= 2:
                    continue
                raise ValueError(
                    "mesh2d does not provide enough active nodes for snap "
                    f"rule span on {rule['axis']}-axis shared rail "
                    f"{rule['rail_id']} at z={z}: "
                    f"[{rule['span_min']}, {rule['span_max']}]"
                )

    def _restore_shared_rail_baseline(self, nodes):
        """Move indexed structural rail nodes back to shared coordinates."""
        nodes = np.asarray(nodes)
        if nodes.ndim != 2 or nodes.shape[1] < 2:
            raise ValueError("nodes must have shape (n, 2+)")
        if not np.issubdtype(nodes.dtype, np.floating):
            raise ValueError("nodes must contain floating point coordinates")

        restored_node_chunks = []
        for axis, coord_axis in (("x", 0), ("y", 1)):
            for rail_index in self.rail_node_index[axis]:
                node_ids = rail_index["node_ids"]
                if not len(node_ids):
                    continue
                if int(node_ids.max()) >= len(nodes):
                    raise ValueError(
                        "nodes do not contain every indexed shared-rail node"
                    )
                changed_mask = nodes[node_ids, coord_axis] != rail_index["coord"]
                nodes[node_ids, coord_axis] = rail_index["coord"]
                if np.any(changed_mask):
                    restored_node_chunks.append(node_ids[changed_mask])

        if not restored_node_chunks:
            return 0
        return int(len(np.unique(np.concatenate(restored_node_chunks))))

    def _restore_current_snap_state(self, nodes):
        """Restore only rail coordinates changed through this mesher."""
        nodes = np.asarray(nodes)
        if nodes.ndim != 2 or nodes.shape[1] < 2:
            raise ValueError("nodes must have shape (n, 2+)")
        if not np.issubdtype(nodes.dtype, np.floating):
            raise ValueError("nodes must contain floating point coordinates")

        restored_node_chunks = []
        for axis, coord_axis in (("x", 0), ("y", 1)):
            node_ids = self._current_changed_node_ids[axis]
            if not len(node_ids):
                continue
            if int(node_ids.max()) >= len(nodes):
                raise ValueError(
                    "nodes do not contain every changed shared-rail node"
                )
            baseline = self._baseline_coords_for_node_ids(axis, node_ids)
            changed_mask = nodes[node_ids, coord_axis] != baseline
            nodes[node_ids, coord_axis] = baseline
            if np.any(changed_mask):
                restored_node_chunks.append(node_ids[changed_mask])

        self._current_changed_node_ids = {
            "x": np.empty(0, dtype=np.intp),
            "y": np.empty(0, dtype=np.intp),
        }
        self._current_changed_node_values = {
            "x": np.empty(0, dtype=np.float64),
            "y": np.empty(0, dtype=np.float64),
        }
        if not restored_node_chunks:
            return 0
        return int(len(np.unique(np.concatenate(restored_node_chunks))))

    def build(
        self,
        obj_list,
        preserve_mesh2d=False,
        allow_independent_bodies=False,
    ):
        """Build a 3D mesh by snapping the 2D rails and dragging each layer.

        Args:
            obj_list: Dragger-compatible object stack data. Each object is a
                list of z layers; every layer except the final sentinel must
                provide ``areas`` and ``element_size``.
            preserve_mesh2d: If true, copy the 2D node coordinates once before
                dragging. The default is zero-copy and mutates
                ``self.mesh2d.nodes`` in place as snap rules are applied.
            allow_independent_bodies: Explicitly allow more than one non-empty
                object stack.  Each stack owns separate 3D nodes, including at
                touching interfaces, so the default rejects this potentially
                non-conformal FEM topology.

        Returns:
            The populated :class:`Dragger` instance.
        """
        for option_name, option_value in (
            ("preserve_mesh2d", preserve_mesh2d),
            ("allow_independent_bodies", allow_independent_bodies),
        ):
            if not isinstance(option_value, (bool, np.bool_)):
                raise ValueError(f"{option_name} must be a boolean")
        preserve_mesh2d = bool(preserve_mesh2d)
        allow_independent_bodies = bool(allow_independent_bodies)
        self._check_mesh_ready()
        # Reject replaced public arrays before dtype normalization can assign a
        # converted replacement back to the caller-owned Mesh2D object.
        self._assert_authoritative_mesh_layout()
        if self.mesh_domain is None:
            raise RuntimeError(
                "build requires an explicit BOX mesh_domain so complete "
                "footprint coverage can be proven"
            )
        if self.mesh_domain["type"] != "BOX":
            raise NotImplementedError(
                "build domain partition validation is currently implemented "
                "only for BOX footprints"
            )
        obj_list = list(obj_list)
        nonempty_stack_count = sum(len(obj) >= 2 for obj in obj_list)
        if nonempty_stack_count > 1 and not allow_independent_bodies:
            raise ValueError(
                "multiple object stacks produce independent non-conformal "
                "3D bodies; combine conformal regions into one stack or pass "
                "allow_independent_bodies=True explicitly"
            )
        self._validate_build_z_events(obj_list)
        self._validate_build_geometry_boundaries(obj_list)
        nodes, elements = self._validate_mesh2d_arrays()
        baseline_overrides = self._managed_baseline_coordinate_overrides(
            nodes,
            require_same_buffer=not preserve_mesh2d,
        )
        self._assert_mesh_integrity(
            nodes,
            elements,
            node_coordinate_overrides=baseline_overrides,
        )
        material_volume_state = self._capture_material_volume_state(obj_list)
        # Construct all independent working objects before rebinding managed
        # snap state, so even allocation/constructor failures leave a preserved
        # caller state untouched.
        dragger_obj = Dragger()

        mesh2d = self.mesh2d
        preserved_snap_state = None
        if preserve_mesh2d:
            # Preserve both the caller-visible coordinates and their managed
            # active-state bookkeeping.  The one requested node copy becomes
            # the private baseline working buffer; no second full copy is
            # needed.
            working_nodes = nodes.copy()
            self._apply_baseline_coordinate_overrides(
                working_nodes,
                baseline_overrides,
            )
            working_mesh2d = Mesh2D(
                nodes=working_nodes,
                elements=elements,
                metadata=getattr(self.mesh2d, "metadata", None),
            )
            empty_changed_ids = {
                "x": np.empty(0, dtype=np.intp),
                "y": np.empty(0, dtype=np.intp),
            }
            empty_changed_values = {
                "x": np.empty(0, dtype=np.float64),
                "y": np.empty(0, dtype=np.float64),
            }
            preserved_snap_state = (
                {
                    axis: node_ids.copy()
                    for axis, node_ids in self._current_changed_node_ids.items()
                },
                {
                    axis: values.copy()
                    for axis, values in (
                        self._current_changed_node_values.items()
                    )
                },
                self._snap_state_nodes,
            )
            self._current_changed_node_ids = empty_changed_ids
            self._current_changed_node_values = empty_changed_values
            self._snap_state_nodes = working_nodes
            mesh2d = working_mesh2d
        else:
            # Integrity has proven that every untracked coordinate is already
            # baseline and that tracked coordinates still equal their exact
            # managed targets.  Only those validated sparse targets are now
            # normalized; an unmanaged rail edit can no longer be hidden.
            self._apply_baseline_coordinate_overrides(nodes, baseline_overrides)
            self._current_changed_node_ids = {
                "x": np.empty(0, dtype=np.intp),
                "y": np.empty(0, dtype=np.intp),
            }
            self._current_changed_node_values = {
                "x": np.empty(0, dtype=np.float64),
                "y": np.empty(0, dtype=np.float64),
            }
            self._snap_state_nodes = nodes

        dragger_ready = False
        try:
            dragger_obj.set_2D(mesh2d)
            dragger_ready = True
            self._reserve_full_domain_build_capacity(
                dragger_obj,
                obj_list,
            )
            for obj in obj_list:
                if len(obj) < 2:
                    continue

                restore_chunks = [
                    ids
                    for ids in self._current_changed_node_ids.values()
                    if len(ids)
                ]
                restored = self.reset_snap_state(
                    nodes=dragger_obj.node_2D,
                    sparse=True,
                )
                if restored and restore_chunks:
                    restored_node_ids = np.unique(
                        np.concatenate(restore_chunks)
                    )
                    dragger_obj._cal_volumns(
                        element_indices=self._incident_element_ids(
                            restored_node_ids
                        )
                    )
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
                        incident_element_ids = self._incident_element_ids(
                            touched_node_ids
                        )
                        self._sync_dragger_current_layer_xy(
                            dragger_obj,
                            touched_node_ids,
                            incident_element_ids=incident_element_ids,
                        )
                        dragger_obj._cal_volumns(
                            element_indices=incident_element_ids
                        )

                    dragger_obj._organize(layer["areas"], layer_index)
                    dragger_obj._drag(layer["element_size"], z_begin, z_end)

                # The sentinel is a real geometric top plane even though it
                # does not own another extrusion interval.  Apply and sync its
                # complete snap state so the final 3D surface cannot remain at
                # the previous shared-rail coordinates.
                touched, touched_node_ids = self.apply_snap_rules_at_z(
                    obj[-1]["z"],
                    nodes=dragger_obj.node_2D,
                    return_touched_node_ids=True,
                )
                if touched:
                    incident_element_ids = self._incident_element_ids(
                        touched_node_ids
                    )
                    self._sync_dragger_current_layer_xy(
                        dragger_obj,
                        touched_node_ids,
                        incident_element_ids=incident_element_ids,
                    )
                    dragger_obj._cal_volumns(
                        element_indices=incident_element_ids
                    )
        except Exception:
            try:
                if dragger_ready:
                    self.reset_snap_state(
                        nodes=dragger_obj.node_2D,
                        sparse=True,
                    )
            finally:
                if preserve_mesh2d:
                    (
                        self._current_changed_node_ids,
                        self._current_changed_node_values,
                        self._snap_state_nodes,
                    ) = preserved_snap_state
                self._restore_material_volume_state(material_volume_state)
            raise

        # The preserved working copy is no longer managed after return.  Put
        # back the exact caller state that existed before the build.
        if preserve_mesh2d:
            (
                self._current_changed_node_ids,
                self._current_changed_node_values,
                self._snap_state_nodes,
            ) = preserved_snap_state
        self.dragger = dragger_obj
        return dragger_obj

    @staticmethod
    def _capture_material_volume_state(obj_list):
        """Capture caller-owned NORMAL volume fields for failure rollback."""
        state = []
        seen = set()
        for obj in obj_list:
            for layer in obj[:-1]:
                areas = layer.get("areas")
                if isinstance(areas, dict):
                    areas = [areas]
                for area in areas or ():
                    for metal in area.get("metals") or ():
                        identity = id(metal)
                        if identity in seen:
                            continue
                        seen.add(identity)
                        state.append(
                            (
                                metal,
                                "volumn" in metal,
                                metal.get("volumn"),
                            )
                        )
        return state

    @staticmethod
    def _restore_material_volume_state(state):
        """Undo caller-dictionary diagnostics written by a failed build."""
        for metal, existed, old_value in state:
            if existed:
                metal["volumn"] = old_value
            else:
                metal.pop("volumn", None)

    def _reserve_full_domain_build_capacity(self, dragger_obj, obj_list):
        """Reserve exact final arrays for simple full-BOX stack builds.

        General material selectors require classification before their active
        element count is known.  When every slab is exactly the full BOX and
        has no holes or metal overrides, however, the count is provable from
        2D topology and z subdivisions.  Reserving once avoids old+new array
        coexistence at repeated growth boundaries for large multi-layer jobs.
        """
        total_subdivisions_by_stack = []
        slab_plans = []
        for obj in obj_list:
            if len(obj) < 2:
                continue
            subdivision_count = 0
            for layer_index, layer in enumerate(obj[:-1]):
                areas = layer.get("areas")
                if isinstance(areas, dict):
                    areas = [areas]
                if not isinstance(areas, (list, tuple)) or len(areas) != 1:
                    return False
                area = areas[0]
                if not self._is_plain_full_box_area(area):
                    return False

                z_begin = float(layer["z"])
                z_end = float(obj[layer_index + 1]["z"])
                element_size = float(layer["element_size"])
                ratio = (z_end - z_begin) / element_size
                if not np.isfinite(ratio):
                    raise OverflowError(
                        "drag interval requires too many subdivisions"
                    )
                drag_num = max(1, int(np.ceil(ratio)))
                actual_element_size = (z_end - z_begin) / drag_num
                subdivision_count += drag_num
                slab_plans.append(
                    (
                        z_begin,
                        z_end,
                        element_size,
                        drag_num,
                        actual_element_size,
                    )
                )
            total_subdivisions_by_stack.append(subdivision_count)

        if not total_subdivisions_by_stack:
            return False

        element_count_2d = len(dragger_obj.element_2D)
        if self._structured_grid_shape is not None:
            active_node_count_2d = len(dragger_obj.node_2D)
        else:
            active_node_count_2d = len(
                np.unique(dragger_obj.element_2D)
            )
        total_hex_count = element_count_2d * sum(
            total_subdivisions_by_stack
        )
        total_node_count = active_node_count_2d * sum(
            subdivisions + 1
            for subdivisions in total_subdivisions_by_stack
        )
        int32_max = np.iinfo(np.int32).max
        if total_hex_count > int32_max:
            raise OverflowError("3D element count exceeds int32 id capacity")
        if total_node_count > int32_max:
            raise OverflowError(
                "3D node count exceeds int32 connectivity capacity"
            )
        # Capacity reservation must never allocate output for a slab that the
        # dragger will deterministically reject.  Prove all float64 z planes
        # only after cheap count bounds, but before either final array exists.
        for slab_plan in slab_plans:
            dragger_obj._validate_drag_z_planes(*slab_plan)
        dragger_obj._pre_allocate_elements(total_hex_count)
        dragger_obj._pre_allocate_nodes(total_node_count)
        return True

    def _is_plain_full_box_area(self, area):
        """Return whether one area provably activates the complete 2D mesh."""
        if not isinstance(area, dict) or area.get("type") != "BOX":
            return False
        if area.get("material") == "EMPTY":
            return False
        if area.get("holes") or area.get("metals"):
            return False
        try:
            x1, y1, x2, y2 = [float(value) for value in area["dim"]]
        except (KeyError, TypeError, ValueError):
            return False
        area_bounds = (
            min(x1, x2),
            min(y1, y2),
            max(x1, x2),
            max(y1, y2),
        )
        return area_bounds == tuple(self.mesh_domain["bbox"])

    def _validate_build_z_events(self, obj_list):
        """Require every stack to split at all pattern lifecycle events."""
        events = self._rule_event_z_values
        for object_index, obj in enumerate(obj_list):
            if len(obj) < 2:
                continue
            z_values = np.asarray(
                [float(layer["z"]) for layer in obj],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(z_values)):
                raise ValueError("build layer z coordinates must be finite")
            z_intervals = np.diff(z_values)
            if np.any(~np.isfinite(z_intervals)):
                raise ValueError(
                    f"object stack {object_index} z intervals must be finite"
                )
            if np.any(z_intervals <= 0.0):
                raise ValueError(
                    f"object stack {object_index} z coordinates must be "
                    "strictly increasing"
                )
            z_set = set(float(value) for value in z_values)
            missing = [
                float(event)
                for event in events
                if z_values[0] < event < z_values[-1]
                and float(event) not in z_set
            ]
            if missing:
                raise ValueError(
                    f"object stack {object_index} crosses pattern z events "
                    f"without layer boundaries: {missing}"
                )

    def _validate_build_geometry_boundaries(self, obj_list):
        """Require every material selector boundary to be an active feature.

        ``Dragger`` selects whole 2D elements.  If an area, range, or hole cuts
        through an element, centroid/corner classification can silently omit
        or leak material.  All selector edges are therefore proven against
        the exact pattern-line union at that layer z before any extrusion or
        mesh mutation begins.  BOX domain edges are permanent boundaries.
        Work is proportional to input geometry, not mesh element count.
        """
        boundary_index = self._active_pattern_boundary_index()
        for object_index, obj in enumerate(obj_list):
            if len(obj) < 2:
                continue
            for layer_index, layer in enumerate(obj[:-1]):
                if "areas" not in layer:
                    raise ValueError(
                        f"object stack {object_index} layer {layer_index} "
                        "must provide areas"
                    )
                self._validate_build_element_size(
                    layer.get("element_size"),
                    object_index,
                    layer_index,
                )
                z_begin = float(layer["z"])
                z_end = float(obj[layer_index + 1]["z"])
                areas = layer["areas"]
                if areas is None:
                    continue
                if isinstance(areas, dict):
                    areas = [areas]
                elif not isinstance(areas, (list, tuple)):
                    raise ValueError("build layer areas must be a mapping or list")

                for area_index, area in enumerate(areas):
                    context = (
                        f"object {object_index} layer {layer_index} "
                        f"area {area_index}"
                    )
                    self._validate_build_face_boundary(
                        area,
                        z_begin,
                        z_end,
                        boundary_index,
                        context,
                    )
                    metals = self._validate_build_material_schema(
                        area,
                        context,
                    )
                    for hole_index, hole in enumerate(area.get("holes") or ()):
                        self._validate_build_face_boundary(
                            hole,
                            z_begin,
                            z_end,
                            boundary_index,
                            f"{context} hole {hole_index}",
                        )
                    for metal_index, metal in enumerate(metals):
                        for selector_name in ("ranges", "holes"):
                            selectors = metal.get(selector_name) or ()
                            if isinstance(selectors, dict):
                                selectors = [selectors]
                            for selector_index, selector in enumerate(selectors):
                                self._validate_build_face_boundary(
                                    selector,
                                    z_begin,
                                    z_end,
                                    boundary_index,
                                    f"{context} metal {metal_index} "
                                    f"{selector_name} {selector_index}",
                                )

    @classmethod
    def _validate_build_material_schema(cls, area, context):
        """Validate all material assignment fields before mesh mutation."""
        cls._validate_material_label(area.get("material"), f"{context} material")
        area_holes = area.get("holes")
        if area_holes is not None and not isinstance(
            area_holes,
            (list, tuple),
        ):
            raise ValueError(f"{context} holes must be a list or tuple")
        metals = area.get("metals")
        if metals is None:
            return ()
        if not isinstance(metals, (list, tuple)):
            raise ValueError(f"{context} metals must be a list or tuple")

        supported_types = {"NORMAL", "CONTINUE", "CONVERT"}
        for metal_index, metal in enumerate(metals):
            metal_context = f"{context} metal {metal_index}"
            if not isinstance(metal, dict):
                raise ValueError(f"{metal_context} must be a mapping")
            metal_type = metal.get("type")
            if metal_type not in supported_types:
                raise ValueError(
                    f"{metal_context} type must be exactly one of "
                    "NORMAL, CONTINUE, or CONVERT"
                )
            cls._validate_material_label(
                metal.get("material"),
                f"{metal_context} material",
            )
            if metal_type == "CONVERT":
                cls._validate_material_label(
                    metal.get("material_o"),
                    f"{metal_context} material_o",
                )
            if metal_type == "NORMAL":
                density = metal.get("density")
                if (
                    isinstance(density, (bool, np.bool_))
                    or not isinstance(
                        density,
                        (int, float, np.integer, np.floating),
                    )
                ):
                    raise ValueError(
                        f"{metal_context} density must be finite and within "
                        "[0, 100]"
                    )
                density = float(density)
                if not np.isfinite(density) or not 0.0 <= density <= 100.0:
                    raise ValueError(
                        f"{metal_context} density must be finite and within "
                        "[0, 100]"
                    )
            for selector_name in ("ranges", "holes"):
                selectors = metal.get(selector_name)
                if selectors is not None and not isinstance(
                    selectors,
                    (list, tuple),
                ):
                    raise ValueError(
                        f"{metal_context} {selector_name} must be a list or "
                        "tuple"
                    )
        return metals

    @staticmethod
    def _validate_material_label(value, context):
        """Require deterministic string component identifiers."""
        if not isinstance(value, str) or not value:
            raise ValueError(f"{context} must be a non-empty string")
        if value == "EMPTY":
            raise ValueError(
                f"{context} must not use the reserved EMPTY component"
            )

    @staticmethod
    def _validate_build_element_size(
        element_size,
        object_index,
        layer_index,
    ):
        """Validate extrusion size before allocating any 3D state."""
        try:
            element_size = float(element_size)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"object stack {object_index} layer {layer_index} "
                "element_size must be positive and finite"
            ) from exc
        if not np.isfinite(element_size) or element_size <= 0.0:
            raise ValueError(
                f"object stack {object_index} layer {layer_index} "
                "element_size must be positive and finite"
            )

    def _active_pattern_boundary_index(self):
        """Index exact pattern spans by axis and fixed coordinate."""
        result = {
            "features": {"x": {}, "y": {}},
            "rail_rules": {"x": {}, "y": {}},
        }
        lines = _extract_lines(self.faces)
        vertical, horizontal = _classify_line(lines)
        for axis, axis_lines in (("x", vertical), ("y", horizontal)):
            for line in axis_lines:
                x1, y1, x2, y2, z_bottom, z_top = _line_components(line)
                if axis == "x":
                    coord = x1
                    span_min, span_max = sorted((y1, y2))
                else:
                    coord = y1
                    span_min, span_max = sorted((x1, x2))
                result["features"][axis].setdefault(float(coord), []).append(
                    (
                        float(span_min),
                        float(span_max),
                        float(z_bottom),
                        float(z_top),
                    )
                )

        if self.mesh_domain is not None and self.mesh_domain["type"] == "BOX":
            xmin, ymin, xmax, ymax = self.mesh_domain["bbox"]
            permanent = (
                ("x", xmin, ymin, ymax),
                ("x", xmax, ymin, ymax),
                ("y", ymin, xmin, xmax),
                ("y", ymax, xmin, xmax),
            )
            for axis, coord, span_min, span_max in permanent:
                result["features"][axis].setdefault(
                    float(coord),
                    [],
                ).append(
                    (float(span_min), float(span_max), None, None)
                )

        for rules in self.snap_rules_by_z.values():
            for rule in rules:
                axis = rule["axis"]
                rail_coord = self._rule_rail_coord(rule)
                result["rail_rules"][axis].setdefault(
                    rail_coord,
                    [],
                ).append(rule)
        return result

    def _validate_build_face_boundary(
        self,
        face,
        z_begin,
        z_end,
        boundary_index,
        context,
    ):
        """Prove every edge of one build selector from active pattern spans."""
        if not isinstance(face, dict):
            raise ValueError(f"{context} must be a face mapping")
        try:
            raw_face_type = face["type"]
            dim = face["dim"]
        except KeyError as exc:
            raise ValueError(f"{context} is missing {exc.args[0]}") from exc
        if raw_face_type not in {"BOX", "POLYGON"}:
            raise ValueError(
                f"{context} type must be exactly BOX or POLYGON"
            )
        face_type = raw_face_type

        canonical_face = {
            "type": face_type,
            "dim": dim,
            "bottom_z": z_begin,
            "top_z": z_end,
        }
        lines = _extract_lines([canonical_face])
        vertical, horizontal = _classify_line(lines)
        for axis, axis_lines in (("x", vertical), ("y", horizontal)):
            for line in axis_lines:
                x1, y1, x2, y2, _, _ = _line_components(line)
                if axis == "x":
                    coord = float(x1)
                    span_min, span_max = sorted((float(y1), float(y2)))
                else:
                    coord = float(y1)
                    span_min, span_max = sorted((float(x1), float(x2)))
                active_intervals = [
                    (available_min, available_max)
                    for (
                        available_min,
                        available_max,
                        z_bottom,
                        z_top,
                    ) in boundary_index["features"][axis].get(coord, ())
                    if z_bottom is None
                    or (z_bottom <= z_begin and z_top >= z_end)
                ]
                active_intervals.extend(
                    self._available_structural_rail_intervals(
                        boundary_index,
                        axis,
                        coord,
                        z_begin,
                        z_end,
                    )
                )
                if not self._interval_union_covers(
                    active_intervals,
                    span_min,
                    span_max,
                ):
                    raise ValueError(
                        f"{context} boundary on {axis}={coord}, span "
                        f"[{span_min}, {span_max}] is not an active exact "
                        f"pattern mesh edge throughout z=[{z_begin}, {z_end}]"
                    )

    def _available_structural_rail_intervals(
        self,
        boundary_index,
        axis,
        coord,
        z_begin,
        z_end,
    ):
        """Return baseline rail spans not displaced in this exact z state."""
        rail_rules = boundary_index["rail_rules"][axis].get(coord, ())
        if not rail_rules:
            return []
        baseline_intervals = [
            (float(rule["span_min"]), float(rule["span_max"]))
            for rule in rail_rules
        ]
        displaced_intervals = [
            (float(rule["span_min"]), float(rule["span_max"]))
            for rule in rail_rules
            if float(rule["target_coord"]) != coord
            and float(rule["z_bottom"]) <= z_end
            and float(rule["z_top"]) >= z_begin
        ]
        if not displaced_intervals:
            return baseline_intervals

        available = baseline_intervals
        for cut_min, cut_max in sorted(displaced_intervals):
            next_available = []
            for span_min, span_max in available:
                # These are closed intervals.  Endpoint contact is still a
                # displaced structural node, so only a strict gap is
                # disjoint.  The retained open-side portions below use
                # nextafter to exclude the displaced endpoint exactly.
                if cut_max < span_min or cut_min > span_max:
                    next_available.append((span_min, span_max))
                    continue
                if span_min < cut_min:
                    left_end = float(np.nextafter(cut_min, -np.inf))
                    if left_end >= span_min:
                        next_available.append((span_min, left_end))
                if cut_max < span_max:
                    right_start = float(np.nextafter(cut_max, np.inf))
                    if right_start <= span_max:
                        next_available.append((right_start, span_max))
            available = next_available
            if not available:
                break
        return available

    @staticmethod
    def _interval_union_covers(intervals, required_min, required_max):
        """Return whether exact closed intervals continuously cover a span."""
        cursor = required_min
        for interval_min, interval_max in sorted(intervals):
            if interval_max < cursor:
                continue
            if interval_min > cursor:
                return False
            cursor = max(cursor, interval_max)
            if cursor >= required_max:
                return True
        return False

    def _check_mesh_ready(self):
        """Raise an error if the 2D mesh and rail lookup are unavailable."""
        if (
            self.mesh2d is None
            or self.rail_node_index is None
            or self._mesh_pattern_generation != self._pattern_generation
        ):
            raise RuntimeError(
                "Error: mesh_checkerboard or "
                "mesh_assignment is not performed"
            )
        self._assert_pattern_integrity()

    def _sync_dragger_current_layer_xy(
        self,
        dragger_obj,
        node_ids,
        incident_element_ids=None,
    ):
        """Sync snapped 2D nodes into already-created top-layer 3D nodes."""
        node_ids = np.asarray(node_ids, dtype=np.intp)
        if node_ids.size == 0:
            return 0

        return dragger_obj.sync_top_plane_xy(
            node_ids,
            incident_element_ids=incident_element_ids,
        )
