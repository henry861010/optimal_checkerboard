import numpy as np
from matplotlib.path import Path

from optimal_checkerboard.algorithms.polygon import normalize_polygon_loops

'''
    OBJECTIVE: 
        1. Used for the 2.5D Auotomation Framework (engine.py)
'''

ELEMENT_LEN = 8
NODE_LEN = 3
_VALIDATION_CHUNK_SIZE = 262_144
_CONNECTIVITY_CHUNK_SIZE = 262_144
_POLYGON_BOUNDARY_MATRIX_ENTRY_BUDGET = 1_048_576
_MAX_ALLOCATION_SLACK = 1_000_000
_SPATIAL_INDEX_MIN_ELEMENTS = 4_096
_SPATIAL_INDEX_MIN_SELECTORS = 8
_SPATIAL_INDEX_BUILD_CHUNK_SIZE = 262_144


def _cross_2d(vector_a, vector_b):
    """Return the row-wise scalar cross product of 2D vectors."""
    return vector_a[:, 0] * vector_b[:, 1] - vector_a[:, 1] * vector_b[:, 0]


def _write_drag_connectivity_chunk(
    destination,
    local_connectivity,
    plane_indices,
    base_map,
    node_start,
    nodes_per_plane,
):
    """Write one bounded chunk in plane-major final-hexahedron order."""
    bottom_offsets = (
        node_start + (plane_indices - 1) * nodes_per_plane
    )
    top_offsets = node_start + plane_indices * nodes_per_plane
    np.add(
        local_connectivity,
        bottom_offsets[:, None],
        out=destination[:, :4],
        casting="unsafe",
    )
    first_plane = plane_indices == 0
    if np.any(first_plane):
        destination[first_plane, :4] = base_map[
            local_connectivity[first_plane]
        ]
    np.add(
        local_connectivity,
        top_offsets[:, None],
        out=destination[:, 4:],
        casting="unsafe",
    )


def _search_polygon_element(x4, y4, dim, eps=0.0):
    loops = normalize_polygon_loops(dim)
    points = np.stack((x4, y4), axis=-1).reshape(-1, 2)
    element_count = len(x4)
    included = np.zeros(element_count, dtype=bool)

    # Evaluate one legal region at a time.  OR-ing point masks from all hulls
    # before checking four corners would incorrectly accept a large element
    # whose corners are split across two disjoint hulls.
    for hull_index, hull in enumerate(loops):
        if hull["role"] != "hull":
            continue
        hull_mask = _points_in_loop_inclusive(
            points,
            hull["points"],
            eps=eps,
        ).reshape(element_count, 4)
        region_mask = hull_mask.all(axis=1)
        if not np.any(region_mask):
            continue

        for hole in loops:
            if (
                hole["role"] != "hole"
                or hole["hull_index"] != hull_index
            ):
                continue
            loop_mask = _points_in_loop_inclusive(
                points,
                hole["points"],
                eps=eps,
            )
            boundary_mask = _points_on_loop_boundary(
                points,
                np.asarray(hole["points"], dtype=float),
                eps=eps,
            )
            strict_inside_mask = (loop_mask & ~boundary_mask).reshape(
                element_count,
                4,
            )
            loop_mask = loop_mask.reshape(element_count, 4)
            region_mask &= ~(
                strict_inside_mask.any(axis=1) | loop_mask.all(axis=1)
            )
        included |= region_mask

    return included


def _points_in_loop_inclusive(points, loop, eps=0.0):
    vertices = np.asarray(loop, dtype=float)
    path_mask = Path(vertices).contains_points(points)
    boundary_mask = _points_on_loop_boundary(points, vertices, eps=eps)
    return path_mask | boundary_mask


def _point_chunk_on_loop_boundary(
    points,
    x1,
    y1,
    x2,
    y2,
    dx,
    dy,
    tol,
    cross_tol,
):
    """Return boundary hits for one bounded point-by-segment matrix."""
    x = points[:, 0:1]
    y = points[:, 1:2]
    cross = (x - x1) * dy - (y - y1) * dx
    within_x = (x >= np.minimum(x1, x2) - tol) & (
        x <= np.maximum(x1, x2) + tol
    )
    within_y = (y >= np.minimum(y1, y2) - tol) & (
        y <= np.maximum(y1, y2) + tol
    )
    return np.any(
        (np.abs(cross) <= cross_tol) & within_x & within_y,
        axis=1,
    )


def _points_on_loop_boundary(points, vertices, eps=0.0, chunk_size=65536):
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")

    tol = _polygon_boundary_tolerance(points, vertices, eps)
    x1 = vertices[:, 0]
    y1 = vertices[:, 1]
    x2 = np.roll(x1, -1)
    y2 = np.roll(y1, -1)
    dx = x2 - x1
    dy = y2 - y1
    cross_tol = tol * np.maximum(np.maximum(np.abs(dx), np.abs(dy)), 1.0)

    # Each point is broadcast against every polygon segment.  Bound matrix
    # entries, rather than just point rows, so a high-vertex polygon cannot
    # silently turn a nominally small point chunk into a huge temporary.
    vertex_count = max(int(len(vertices)), 1)
    bounded_chunk_size = max(
        1,
        min(
            int(chunk_size),
            _POLYGON_BOUNDARY_MATRIX_ENTRY_BUDGET // vertex_count,
        ),
    )
    result = np.zeros(len(points), dtype=bool)
    for start in range(0, len(points), bounded_chunk_size):
        end = min(start + bounded_chunk_size, len(points))
        chunk = points[start:end]
        result[start:end] = _point_chunk_on_loop_boundary(
            chunk,
            x1,
            y1,
            x2,
            y2,
            dx,
            dy,
            tol,
            cross_tol,
        )
    return result


def _polygon_boundary_tolerance(points, vertices, eps):
    if eps:
        return float(eps)
    # Input floats are authoritative topology.  A scale-derived tolerance can
    # swallow a real thin element at large coordinates and leak material
    # across a polygon boundary.
    return 0.0


def _selector_bbox(selector):
    """Return a validated conservative bbox for an indexable selector.

    ``None`` means that candidate pruning is not safe.  The caller then uses
    the existing full scan, which remains the authority for validation and
    inclusion semantics.
    """
    if not isinstance(selector, dict):
        return None
    selector_type = selector.get("type")
    dim = selector.get("dim")
    if selector_type == "BOX":
        try:
            if len(dim) != 4:
                return None
            x1, y1, x2, y2 = [float(value) for value in dim]
        except (TypeError, ValueError, OverflowError):
            return None
        if not np.all(np.isfinite((x1, y1, x2, y2))):
            return None
        min_x, max_x = sorted((x1, x2))
        min_y, max_y = sorted((y1, y2))
        if min_x >= max_x or min_y >= max_y:
            return None
        return min_x, min_y, max_x, max_y

    if selector_type == "POLYGON":
        try:
            loops = normalize_polygon_loops(dim)
        except (TypeError, ValueError, OverflowError):
            return None
        hull_points = [
            point
            for loop in loops
            if loop["role"] == "hull"
            for point in loop["points"]
        ]
        points = np.asarray(hull_points, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1] != 2
            or not np.all(np.isfinite(points))
        ):
            return None
        return (
            float(points[:, 0].min()),
            float(points[:, 1].min()),
            float(points[:, 0].max()),
            float(points[:, 1].max()),
        )
    return None


def _element_anchor_chunk(node_2d, element_2d, start, stop):
    """Return an actual quad corner, never an arithmetic approximation."""
    return node_2d[element_2d[start:stop, 0]]


class _ElementAnchorBinIndex:
    """One-organize uniform-bin index over one current quad corner.

    A selector bbox is only a conservative prefilter.  Every returned element
    still passes through ``search_face_element`` before it can be assigned.
    """

    def __init__(
        self,
        element_count,
        min_x,
        min_y,
        max_x,
        max_y,
        x_bin_count,
        y_bin_count,
        offsets,
        ordered_element_ids,
    ):
        self.element_count = int(element_count)
        self.min_x = float(min_x)
        self.min_y = float(min_y)
        self.max_x = float(max_x)
        self.max_y = float(max_y)
        self.x_bin_count = int(x_bin_count)
        self.y_bin_count = int(y_bin_count)
        self.offsets = offsets
        self.ordered_element_ids = ordered_element_ids

    @classmethod
    def build(
        cls,
        node_2d,
        element_2d,
        selector_count,
        chunk_size=_SPATIAL_INDEX_BUILD_CHUNK_SIZE,
    ):
        element_count = len(element_2d)
        if (
            element_count == 0
            or element_count > np.iinfo(np.int32).max
            or selector_count <= 0
        ):
            return None
        if not np.all(np.isfinite(node_2d)):
            # The original full scan would expose a non-finite referenced
            # corner.  Falling back preserves that fail-closed behavior even
            # when the corrupt element is outside every selector bbox.
            return None

        min_x = np.inf
        min_y = np.inf
        max_x = -np.inf
        max_y = -np.inf
        for start in range(0, element_count, int(chunk_size)):
            stop = min(start + int(chunk_size), element_count)
            anchors = _element_anchor_chunk(
                node_2d,
                element_2d,
                start,
                stop,
            )
            # A stale or externally corrupted node buffer must not become
            # invisible merely because its element falls outside a bbox.
            if not np.all(np.isfinite(anchors)):
                return None
            min_x = min(min_x, float(anchors[:, 0].min()))
            min_y = min(min_y, float(anchors[:, 1].min()))
            max_x = max(max_x, float(anchors[:, 0].max()))
            max_y = max(max_y, float(anchors[:, 1].max()))

        span_x = max_x - min_x
        span_y = max_y - min_y
        if not np.all(np.isfinite((span_x, span_y))):
            return None

        # With at most about 1,000 geometry selectors, S^2 bins give narrow
        # strips enough resolution to keep aggregate exact-predicate work near
        # E plus boundary-bin spill.  The cap keeps index storage O(E).
        target_bin_count = min(
            element_count,
            max(256, int(selector_count) * int(selector_count)),
        )
        if span_x == 0.0:
            x_bin_count = 1
            y_bin_count = target_bin_count
        elif span_y == 0.0:
            x_bin_count = target_bin_count
            y_bin_count = 1
        else:
            scale = max(span_x, span_y)
            normalized_x = span_x / scale
            normalized_y = span_y / scale
            if normalized_x == 0.0:
                x_bin_count = 1
                y_bin_count = target_bin_count
            elif normalized_y == 0.0:
                x_bin_count = target_bin_count
                y_bin_count = 1
            else:
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    raw_x_bins = np.sqrt(
                        target_bin_count * normalized_x / normalized_y
                    )
                if not np.isfinite(raw_x_bins):
                    # An extreme but finite aspect ratio can overflow only the
                    # bin-layout heuristic.  Candidate pruning is optional, so
                    # retain exact behavior through the bounded full scan.
                    return None
                x_bin_count = min(
                    target_bin_count,
                    max(1, int(round(float(raw_x_bins)))),
                )
                y_bin_count = min(
                    target_bin_count,
                    max(1, int(np.ceil(target_bin_count / x_bin_count))),
                )
        bin_count = x_bin_count * y_bin_count
        if bin_count > np.iinfo(np.int32).max:
            return None

        bin_ids = np.empty(element_count, dtype=np.int32)
        for start in range(0, element_count, int(chunk_size)):
            stop = min(start + int(chunk_size), element_count)
            anchors = _element_anchor_chunk(
                node_2d,
                element_2d,
                start,
                stop,
            )
            if x_bin_count == 1:
                x_bins = np.zeros(stop - start, dtype=np.int32)
            else:
                x_bins = np.floor(
                    ((anchors[:, 0] - min_x) / span_x) * x_bin_count
                ).astype(np.int32)
                np.clip(x_bins, 0, x_bin_count - 1, out=x_bins)
            if y_bin_count == 1:
                y_bins = np.zeros(stop - start, dtype=np.int32)
            else:
                y_bins = np.floor(
                    ((anchors[:, 1] - min_y) / span_y) * y_bin_count
                ).astype(np.int32)
                np.clip(y_bins, 0, y_bin_count - 1, out=y_bins)
            bin_ids[start:stop] = y_bins * x_bin_count + x_bins

        counts = np.bincount(bin_ids, minlength=bin_count)
        offsets = np.empty(bin_count + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        ordered_element_ids = np.argsort(
            bin_ids,
            kind="stable",
        ).astype(np.int32, copy=False)
        return cls(
            element_count,
            min_x,
            min_y,
            max_x,
            max_y,
            x_bin_count,
            y_bin_count,
            offsets,
            ordered_element_ids,
        )

    @staticmethod
    def _axis_bin_range(low, high, axis_min, axis_max, bin_count):
        if high < axis_min or low > axis_max:
            return None
        if bin_count == 1 or axis_min == axis_max:
            return 0, 0

        # Outward one-ULP expansion conservatively includes selector and bin
        # boundaries.  It can only add spill candidates.
        low = max(float(np.nextafter(low, -np.inf)), axis_min)
        high = min(float(np.nextafter(high, np.inf)), axis_max)
        span = axis_max - axis_min
        low_bin = int(np.floor(((low - axis_min) / span) * bin_count))
        high_bin = int(np.floor(((high - axis_min) / span) * bin_count))
        return (
            max(0, min(bin_count - 1, low_bin)),
            max(0, min(bin_count - 1, high_bin)),
        )

    def _bbox_bin_rectangle(self, bbox):
        min_x, min_y, max_x, max_y = bbox
        x_range = self._axis_bin_range(
            min_x,
            max_x,
            self.min_x,
            self.max_x,
            self.x_bin_count,
        )
        y_range = self._axis_bin_range(
            min_y,
            max_y,
            self.min_y,
            self.max_y,
            self.y_bin_count,
        )
        if x_range is None or y_range is None:
            return None
        return x_range[0], y_range[0], x_range[1], y_range[1]

    def _elements_for_bin_ids(self, bin_ids):
        if len(bin_ids) == 0:
            return np.empty(0, dtype=np.int32)

        # Adjacent bins are adjacent in the CSR element order.  Coalescing
        # them bounds Python objects by bin runs, while the output is at most E.
        run_starts = np.concatenate(
            (
                np.array([0], dtype=np.intp),
                np.flatnonzero(bin_ids[1:] != bin_ids[:-1] + 1) + 1,
            )
        )
        run_stops = np.concatenate(
            (run_starts[1:], np.array([len(bin_ids)], dtype=np.intp))
        )
        output_size = 0
        for run_start, run_stop in zip(run_starts, run_stops):
            first_bin = int(bin_ids[run_start])
            last_bin = int(bin_ids[run_stop - 1])
            output_size += int(
                self.offsets[last_bin + 1] - self.offsets[first_bin]
            )
        if output_size == self.element_count:
            return np.arange(self.element_count, dtype=np.int32)

        candidates = np.empty(output_size, dtype=np.int32)
        output_start = 0
        for run_start, run_stop in zip(run_starts, run_stops):
            first_bin = int(bin_ids[run_start])
            last_bin = int(bin_ids[run_stop - 1])
            ordered_start = int(self.offsets[first_bin])
            ordered_stop = int(self.offsets[last_bin + 1])
            output_stop = output_start + ordered_stop - ordered_start
            candidates[output_start:output_stop] = self.ordered_element_ids[
                ordered_start:ordered_stop
            ]
            output_start = output_stop
        candidates.sort()
        return candidates

    def candidate_local_indices(self, ranges, element_indices):
        """Return sorted local candidates, or ``None`` for safe fallback."""
        if not isinstance(ranges, (list, tuple)) or not ranges:
            return None
        bboxes = []
        for selector in ranges:
            bbox = _selector_bbox(selector)
            if bbox is None:
                return None
            bboxes.append(bbox)

        if element_indices is not None:
            element_indices = np.asarray(element_indices)
            if (
                element_indices.ndim != 1
                or not np.issubdtype(element_indices.dtype, np.integer)
                or (
                    len(element_indices) > 1
                    and np.any(element_indices[1:] <= element_indices[:-1])
                )
            ):
                return None

        rectangles = [
            rectangle
            for bbox in bboxes
            for rectangle in [self._bbox_bin_rectangle(bbox)]
            if rectangle is not None
        ]
        if not rectangles:
            return np.empty(0, dtype=np.int32)

        bin_count = self.x_bin_count * self.y_bin_count
        if len(rectangles) == 1:
            min_x, min_y, max_x, max_y = rectangles[0]
            width = max_x - min_x + 1
            bin_ids = np.empty(
                width * (max_y - min_y + 1),
                dtype=np.int32,
            )
            output_start = 0
            for y_bin in range(min_y, max_y + 1):
                row_start = y_bin * self.x_bin_count + min_x
                output_stop = output_start + width
                bin_ids[output_start:output_stop] = np.arange(
                    row_start,
                    row_start + width,
                    dtype=np.int32,
                )
                output_start = output_stop
        else:
            selected_bins = np.zeros(bin_count, dtype=bool)
            selected_bins_2d = selected_bins.reshape(
                self.y_bin_count,
                self.x_bin_count,
            )
            for min_x, min_y, max_x, max_y in rectangles:
                selected_bins_2d[
                    min_y : max_y + 1,
                    min_x : max_x + 1,
                ] = True
            bin_ids = np.flatnonzero(selected_bins).astype(
                np.int32,
                copy=False,
            )

        global_candidates = self._elements_for_bin_ids(bin_ids)
        if element_indices is None:
            return global_candidates
        if len(global_candidates) == 0 or len(element_indices) == 0:
            return np.empty(0, dtype=np.int32)

        local_positions = np.searchsorted(element_indices, global_candidates)
        in_subset = local_positions < len(element_indices)
        matching_positions = local_positions[in_subset]
        matching_candidates = global_candidates[in_subset]
        matches = (
            element_indices[matching_positions] == matching_candidates
        )
        return matching_positions[matches].astype(np.int32, copy=False)


def search_face_element(element_coordinates, type, dim, index=None, eps=0.0, returnMask=False):
    """
        Fast predicate on subset indices (index). 
        Returns a boolean mask aligned to index (or to all rows if index is None).
    """
    try:
        element_coordinates = np.asarray(
            element_coordinates,
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "element_coordinates must be a numeric (n, 8) array"
        ) from exc
    if element_coordinates.ndim != 2 or element_coordinates.shape[1] != 8:
        raise ValueError("element_coordinates must have shape (n, 8)")
    if not np.all(np.isfinite(element_coordinates)):
        raise ValueError("element_coordinates must be finite")

    try:
        eps = float(eps)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "face search eps must be finite and non-negative"
        ) from exc
    if not np.isfinite(eps) or eps < 0.0:
        raise ValueError("face search eps must be finite and non-negative")
    if index is None:
        selected_coordinates = element_coordinates
    else:
        rows = np.asarray(index)
        if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
            raise ValueError("face search index must contain integer row ids")
        if rows.size and (
            int(rows.min()) < 0 or int(rows.max()) >= len(element_coordinates)
        ):
            raise ValueError("face search index contains out-of-range row ids")
        selected_coordinates = element_coordinates[rows]

    # gather 4 corners
    x4 = selected_coordinates[:, [0, 2, 4, 6]]
    y4 = selected_coordinates[:, [1, 3, 5, 7]]
    
    ### max/min of each element_coordinates
    min_x = x4.min(axis=1)
    max_x = x4.max(axis=1)
    min_y = y4.min(axis=1)
    max_y = y4.max(axis=1)

    if type == "BOX":
        if len(dim) != 4:
            raise ValueError("BOX dim must be [xmin, ymin, xmax, ymax]")
        x1, y1, x2, y2 = [float(value) for value in dim]
        if not np.all(np.isfinite((x1, y1, x2, y2))):
            raise ValueError("BOX coordinates must be finite")
        bl_x, tr_x = sorted((x1, x2))
        bl_y, tr_y = sorted((y1, y2))
        if bl_x >= tr_x or bl_y >= tr_y:
            raise ValueError("BOX dim must have positive area")
        if eps:
            bl_x -= eps
            bl_y -= eps
            tr_x += eps
            tr_y += eps
        res_mask = (min_x >= bl_x) & (max_x <= tr_x) & (min_y >= bl_y) & (max_y <= tr_y)

    elif type == "CYLINDER":
        if len(dim) != 3:
            raise ValueError("CYLINDER dim must be [cx, cy, radius]")
        cx, cy, r = [float(value) for value in dim]
        if not np.all(np.isfinite((cx, cy, r))):
            raise ValueError("CYLINDER coordinates must be finite")
        if r <= 0.0:
            raise ValueError("CYLINDER radius must be positive")
        radius = r + eps if eps else r
        # ``hypot`` scales its operands and stays finite for coordinates where
        # squaring would overflow to inf and make ``inf <= inf`` look inside.
        dist = np.hypot(x4 - cx, y4 - cy)
        res_mask = np.all(dist <= radius, axis=1)
    
    elif type == "POLYGON":
        res_mask = _search_polygon_element(x4, y4, dim, eps=eps)
    else:
        raise ValueError(f"Unsupported type: {type}")
    
    if returnMask:
        return res_mask
    else:
        return np.flatnonzero(res_mask) 
    
class Dragger:
    def __init__(self):
        ### component
        self.comps = {"EMPTY":0}
        
        ### 3D elements
        self.element_num = 0
        self.elements = np.empty((0, ELEMENT_LEN), dtype=np.int32)
        self.element_ids = np.empty((0), dtype=np.int32)
        self.element_comps = np.empty((0), dtype=np.int32)
        
        ### nodes
        self.node_num = 0
        self.nodes = np.empty((0, NODE_LEN), dtype=np.float64)
        self.node_ids = np.empty((0), dtype=np.int32)
        
        ### process
        self.element_2D = np.zeros((0, 4), dtype=np.int32)
        self.element_2D_volumn = np.empty((0), dtype=np.float64)
        self.element_2D_comp = np.empty((0), dtype=np.int32)
        self.previous_element_2D_comp = np.empty((0), dtype=np.int32)
        self._element_2D_active_indices = np.empty(0, dtype=np.int32)
        self._previous_element_2D_active_indices = np.empty(0, dtype=np.int32)
        
        self.node_2D = np.empty((0, 2), dtype=np.float64)
        self.node_2D_to_3D = np.zeros((0), dtype=np.int32)
        self._mapped_node_2D_indices = np.empty(0, dtype=np.int32)
        self._latest_top_hex_by_2D_element = np.empty(0, dtype=np.int32)
        self._node_2D_element_offsets = None
        self._node_2D_element_ids = None
        self._area_claim_stamps = np.empty(0, dtype=np.uint32)
        self._area_claim_generation = 0
        self._selector_spatial_index = None
        
    ### initial
    def set_2D(self, mesh2D:'Mesh2D'):
        if hasattr(mesh2D, "get_byIndex"):
            nodes, elements = mesh2D.get_byIndex()
        else:
            nodes, elements = mesh2D.nodes, mesh2D.elements

        try:
            nodes = np.asarray(nodes, dtype=np.float64)
            raw_elements = np.asarray(elements)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("mesh2D arrays must be rectangular numeric arrays") from exc
        if nodes.ndim != 2 or nodes.shape[1] < 2:
            raise ValueError("mesh2D.nodes must have shape (n, 2+)") 
        if raw_elements.ndim != 2 or raw_elements.shape[1] != 4:
            raise ValueError("mesh2D.elements must have shape (m, 4)")
        if (
            not np.issubdtype(raw_elements.dtype, np.integer)
            or np.issubdtype(raw_elements.dtype, np.bool_)
        ):
            raise ValueError("mesh2D.elements must contain integer node ids")
        if not np.all(np.isfinite(nodes[:, :2])):
            raise ValueError("mesh2D.nodes must contain finite xy coordinates")
        int32_max = np.iinfo(np.int32).max
        if len(nodes) - 1 > int32_max:
            raise OverflowError(
                "mesh2D node count exceeds int32 connectivity capacity"
            )
        if len(raw_elements) > int32_max:
            raise OverflowError("mesh2D element count exceeds int32 index capacity")
        if raw_elements.size:
            min_node_id = int(raw_elements.min())
            max_node_id = int(raw_elements.max())
            if max_node_id > int32_max:
                raise OverflowError(
                    "mesh2D.elements exceed int32 connectivity capacity"
                )
            if min_node_id < 0 or max_node_id >= len(nodes):
                raise ValueError("mesh2D.elements contain out-of-range node ids")
        elements = raw_elements.astype(np.int32, copy=False)
        
        self.element_2D = elements
        self.element_2D_comp = np.zeros(len(elements), dtype=np.int32)
        self.previous_element_2D_comp = np.zeros(len(elements), dtype=np.int32)
        self._element_2D_active_indices = np.empty(0, dtype=np.int32)
        self._previous_element_2D_active_indices = np.empty(0, dtype=np.int32)
        self.element_2D_volumn = np.empty(len(elements), dtype=np.float64)
        self.node_2D = nodes[:,:2]
        self.node_2D_to_3D = np.zeros(len(nodes), dtype=np.int32) - 1
        self._mapped_node_2D_indices = np.empty(0, dtype=np.int32)
        self._latest_top_hex_by_2D_element = (
            np.zeros(len(elements), dtype=np.int32) - 1
        )
        self._node_2D_element_offsets = None
        self._node_2D_element_ids = None
        self._area_claim_stamps = np.zeros(len(elements), dtype=np.uint32)
        self._area_claim_generation = 0
        self._selector_spatial_index = None
        
        self._cal_volumns()
        
    ### foundmental
    def _pre_allocate_nodes(self, size: int = 1):
        if size < 0:
            raise ValueError("node allocation size must be non-negative")
        required = self.node_num + size
        current_capacity = len(self.nodes)
        if required > current_capacity:
            grown_capacity = max(required, int(current_capacity * 1.5))
            new_capacity = min(
                grown_capacity,
                required + _MAX_ALLOCATION_SLACK,
            )
            extra = new_capacity - current_capacity
            
            self.nodes = np.vstack([self.nodes, np.empty((extra, 3), dtype=np.float64)])
            self.node_ids = np.concatenate([self.node_ids, np.empty(extra, dtype=np.int32)])

    def _pre_allocate_elements(self, size: int = 1):
        if size < 0:
            raise ValueError("element allocation size must be non-negative")
        required = self.element_num + size
        current_capacity = len(self.elements)
        if required > current_capacity:
            grown_capacity = max(required, int(current_capacity * 1.5))
            new_capacity = min(
                grown_capacity,
                required + _MAX_ALLOCATION_SLACK,
            )
            extra = new_capacity - current_capacity
            
            self.elements = np.vstack([self.elements, np.empty((extra, 8), dtype=np.int32)])
            self.element_ids = np.concatenate([self.element_ids, np.empty(extra, dtype=np.int32)])
            self.element_comps = np.concatenate([self.element_comps, np.empty(extra, dtype=np.int32)])
        
    ### core
    def _normalize_element_indices(self, element_indices=None):
        if element_indices is None:
            return np.arange(len(self.element_2D), dtype=np.int32)

        element_indices = np.asarray(element_indices, dtype=np.int32)
        if element_indices.ndim == 0:
            element_indices = element_indices.reshape(1)
        return element_indices

    def _element_coordinates(self, element_indices=None):
        element_indices = self._normalize_element_indices(element_indices)
        if len(element_indices) == 0:
            return np.empty((0, 8), dtype=self.node_2D.dtype)

        corner_xy = self.node_2D[self.element_2D[element_indices]]
        return corner_xy.reshape(len(element_indices), 8)

    def _search_faces(
        self,
        element_indices=None,
        ranges=None,
        holes=None,
        returnMask=False,
        chunk_size=_VALIDATION_CHUNK_SIZE,
    ):
        """
        Ranges-first progressive search using only index arrays (no big copies).
        Returns local indices over 'element_indices'. If element_indices is
        None, the returned indices are global 2D element indices.
        """
        if not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        if element_indices is None:
            normalized_indices = None
            n = len(self.element_2D)
        else:
            normalized_indices = self._normalize_element_indices(
                element_indices
            )
            n = len(normalized_indices)
            if n and (
                int(normalized_indices.min()) < 0
                or int(normalized_indices.max()) >= len(self.element_2D)
            ):
                raise ValueError("element_indices contain out-of-range ids")
        if n == 0:
            if returnMask:
                return np.zeros(0, dtype=bool)
            else:
                return np.zeros(0, dtype=np.int32)

        full_mask = np.zeros(n, dtype=bool) if returnMask else None
        candidate_local_indices = None
        if self._selector_spatial_index is not None:
            candidate_local_indices = (
                self._selector_spatial_index.candidate_local_indices(
                    ranges,
                    normalized_indices,
                )
            )
        hit_chunks = []
        scan_count = (
            n
            if candidate_local_indices is None
            else len(candidate_local_indices)
        )
        for start in range(0, scan_count, int(chunk_size)):
            stop = min(start + int(chunk_size), scan_count)
            if candidate_local_indices is None:
                local_indices = np.arange(start, stop, dtype=np.int32)
            else:
                local_indices = candidate_local_indices[start:stop]
            if normalized_indices is None:
                global_indices = local_indices
            else:
                global_indices = normalized_indices[local_indices]
            element_coordinates = self._element_coordinates(global_indices)
            included_mask = self._search_face_coordinate_chunk(
                element_coordinates,
                ranges,
                holes,
            )
            if returnMask:
                full_mask[local_indices] = included_mask
            elif np.any(included_mask):
                hit_chunks.append(
                    local_indices[included_mask].astype(np.int32, copy=False)
                )

        if returnMask:
            return full_mask
        if not hit_chunks:
            return np.empty(0, dtype=np.int32)
        return np.concatenate(hit_chunks).astype(np.int32, copy=False)

    @staticmethod
    def _search_face_coordinate_chunk(element_coordinates, ranges, holes):
        """Return selector hits for one bounded element-coordinate chunk."""
        n = len(element_coordinates)
        included_mask = np.zeros(n, dtype=bool)

        if ranges:
            candidate_indices = np.arange(n, dtype=np.int32)
            for face_range in ranges:
                if not len(candidate_indices):
                    break
                submask = search_face_element(
                    element_coordinates,
                    face_range["type"],
                    face_range["dim"],
                    index=candidate_indices,
                    returnMask=True,
                )
                if np.any(submask):
                    hit_indices = candidate_indices[submask]
                    included_mask[hit_indices] = True
                    candidate_indices = candidate_indices[~submask]
        else:
            included_mask[:] = True

        if holes:
            live_indices = np.flatnonzero(included_mask).astype(
                np.int32,
                copy=False,
            )
            for hole in holes:
                if not len(live_indices):
                    break
                submask = search_face_element(
                    element_coordinates,
                    hole["type"],
                    hole["dim"],
                    index=live_indices,
                    returnMask=True,
                )
                if np.any(submask):
                    lose_indices = live_indices[submask]
                    included_mask[lose_indices] = False
                    live_indices = live_indices[~submask]
        return included_mask
        
    def _assign_metal(self, volumes, density, total_volume, randomSeed=1):
        """
        Randomly pick elements until reaching density% of total_volume.
        Operates by shuffling indices only and using cumsum to avoid Python loops.
        Returns the chosen *row indices within this subset* (not global IDs).
        """
        try:
            density = float(density)
            total_volume = float(total_volume)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "metal density and selector volume must be finite"
            ) from exc
        if (
            not np.isfinite(density)
            or not np.isfinite(total_volume)
        ):
            raise ValueError("metal density and selector volume must be finite")
        if density <= 0:
            return np.empty((0), dtype=np.int32)
            
        volumes = np.asarray(volumes, dtype=np.float64)
        if len(volumes) == 0:
            return np.empty((0), dtype=np.int32)
        if np.any(~np.isfinite(volumes)) or np.any(volumes <= 0.0):
            raise ValueError("metal candidate volumes must be positive and finite")

        target_indices = np.arange(len(volumes), dtype=np.int32)
        
        # target volume
        target = (density / 100.0) * total_volume
        if target <= 0:
            return np.empty((0), dtype=np.int32)

        # random order of candidates (indices only, not rows)
        rng = np.random.default_rng(randomSeed)
        random_indices = target_indices[rng.permutation(len(volumes))]

        # cumulative sum until target
        ordered_volumes = volumes[random_indices]
        with np.errstate(over="ignore", invalid="ignore"):
            csum = np.cumsum(ordered_volumes)
        if not np.all(np.isfinite(csum)):
            # The validated total is finite, so a positive sequential sum can
            # only overflow because its order approaches float64.max.  Scale
            # all terms and the target by the same positive value; this keeps
            # the threshold comparison equivalent without overflow.
            scale = float(ordered_volumes.max())
            csum = np.cumsum(ordered_volumes / scale)
            target = target / scale
        if not np.all(np.isfinite(csum)) or not np.isfinite(target):
            raise OverflowError("metal volume accumulation is not representable")
        chosen_count = min(
            int(np.searchsorted(csum, target, side="left")) + 1,
            len(random_indices),
        )
        return random_indices[:chosen_count]

    @staticmethod
    def _indexable_selector_count(areas):
        def count_selectors(selectors):
            if not isinstance(selectors, (list, tuple)):
                return 0
            return sum(
                isinstance(selector, dict)
                and selector.get("type") in {"BOX", "POLYGON"}
                for selector in selectors
            )

        selector_count = 0
        for area in areas:
            if isinstance(area, dict) and area.get("type") in {
                "BOX",
                "POLYGON",
            }:
                selector_count += 1
            if not isinstance(area, dict):
                continue
            # Holes do not narrow range candidates, but they still contribute
            # to the decision that selector count justifies one shared index.
            selector_count += count_selectors(area.get("holes"))
            for metal in area.get("metals") or ():
                ranges = metal.get("ranges") if isinstance(metal, dict) else None
                if not isinstance(metal, dict):
                    continue
                selector_count += count_selectors(ranges)
                selector_count += count_selectors(metal.get("holes"))
        return selector_count

    def _build_selector_spatial_index(self, selector_count):
        return _ElementAnchorBinIndex.build(
            self.node_2D,
            self.element_2D,
            selector_count,
        )

    def _organize(self, areas, layer=1):
        """Build and discard one current-node selector index per layer."""
        if isinstance(areas, dict):
            normalized_areas = [areas]
        elif areas is None:
            normalized_areas = []
        else:
            normalized_areas = list(areas)

        self._reject_reserved_material_labels(normalized_areas)

        selector_count = self._indexable_selector_count(normalized_areas)
        self._selector_spatial_index = None
        if (
            len(self.element_2D) >= _SPATIAL_INDEX_MIN_ELEMENTS
            and selector_count >= _SPATIAL_INDEX_MIN_SELECTORS
        ):
            self._selector_spatial_index = (
                self._build_selector_spatial_index(selector_count)
            )
        try:
            return self._organize_with_current_index(
                normalized_areas,
                layer=layer,
            )
        finally:
            # node_2D can be snapped before the next z event.  Never retain a
            # anchor index across that mutation boundary.
            self._selector_spatial_index = None

    @staticmethod
    def _reject_reserved_material_labels(areas):
        """Prevent component id 0 from silently deleting selected elements."""
        for area in areas:
            if not isinstance(area, dict):
                continue
            if area.get("material") == "EMPTY":
                raise ValueError("area material must not use reserved EMPTY")
            for metal in area.get("metals") or ():
                if not isinstance(metal, dict):
                    continue
                if metal.get("material") == "EMPTY":
                    raise ValueError("metal material must not use reserved EMPTY")
                if metal.get("material_o") == "EMPTY":
                    raise ValueError(
                        "metal material_o must not use reserved EMPTY"
                    )

    def _organize_with_current_index(self, areas, layer=1):
        """Overlay one layer on the material state of the preceding slab.

        Areas are patches, not a complete replacement cross-section.  Cells
        outside this layer's areas inherit their preceding material.  The
        separate ``previous_element_2D_comp`` buffer remains an immutable
        snapshot while CONTINUE and CONVERT are evaluated.
        """
        if len(self.previous_element_2D_comp) != len(self.element_2D_comp):
            self.previous_element_2D_comp = np.zeros_like(self.element_2D_comp)
            self._previous_element_2D_active_indices = np.empty(
                0,
                dtype=np.int32,
            )

        # Swap fixed-size buffers, clear only stale non-empty cells in the
        # reusable buffer, then sparsely inherit the preceding slab.  Copying
        # the active footprint is output-proportional: every inherited cell
        # will produce at least one hexahedron in the slab that follows.
        previous_active = self._element_2D_active_indices
        reusable_active = self._previous_element_2D_active_indices
        self.previous_element_2D_comp, self.element_2D_comp = (
            self.element_2D_comp,
            self.previous_element_2D_comp,
        )
        self._previous_element_2D_active_indices = previous_active
        if len(reusable_active):
            self.element_2D_comp[reusable_active] = self.comps["EMPTY"]

        inherited_full_footprint = (
            len(previous_active) == len(self.element_2D_comp)
        )
        if inherited_full_footprint:
            self.element_2D_comp[:] = self.previous_element_2D_comp
        elif len(previous_active):
            self.element_2D_comp[previous_active] = (
                self.previous_element_2D_comp[previous_active]
            )

        self._element_2D_active_indices = np.empty(0, dtype=np.int32)
        active_chunks = [] if inherited_full_footprint else [previous_active]
        self._area_claim_generation += 1
        if self._area_claim_generation >= np.iinfo(np.uint32).max:
            self._area_claim_stamps.fill(0)
            self._area_claim_generation = 1
        claim_generation = self._area_claim_generation

        for area in areas:
            ### Select the area once (mask -> indices)
            ranges = [{"type": area["type"], "dim": area["dim"]}]
            holes  = area.get("holes")
            area_indices  = self._search_faces(None, ranges, holes)
            if len(area_indices) == 0:
                continue
            if np.any(
                self._area_claim_stamps[area_indices] == claim_generation
            ):
                raise ValueError(
                    "build areas overlap on one or more 2D elements; express "
                    "priority regions with explicit holes"
                )
            self._area_claim_stamps[area_indices] = claim_generation
            if not inherited_full_footprint:
                active_chunks.append(area_indices)

            # Each selector is evaluated exactly once, in its priority pass,
            # and consumed immediately.  Keeping all selector hit arrays would
            # require O(number_of_metals * area_elements) memory.
            metals = area.get("metals") or ()
            # A one-byte local mask supports O(hits) removals.  Repeated
            # setdiff1d/isin calls sort or rescan the whole remaining pool.
            remaining_mask = np.ones(len(area_indices), dtype=bool)

            ### metal assignment CONTINUE
            for metal in metals:
                if metal["type"] == "CONTINUE":
                    region_indices = self._search_faces(
                        area_indices,
                        metal.get("ranges"),
                        metal.get("holes"),
                    )
                    remaining_target_indices = region_indices[
                        remaining_mask[region_indices]
                    ]
                
                    ### remove the assignment
                    if len(remaining_target_indices) and metal["material"] in self.comps:
                        comp_id = self.comps[metal["material"]]
                        assigned_mask = (
                            self.previous_element_2D_comp[
                                area_indices[remaining_target_indices]
                            ]
                            == comp_id
                        )
                        if np.any(assigned_mask):
                            remaining_assigned_indices = remaining_target_indices[assigned_mask]
                            self.element_2D_comp[
                                area_indices[remaining_assigned_indices]
                            ] = comp_id
                            remaining_mask[remaining_assigned_indices] = False
            
            ### metal assignment CONVERT
            for metal in metals:
                if metal["type"] == "CONVERT":   
                    region_indices = self._search_faces(
                        area_indices,
                        metal.get("ranges"),
                        metal.get("holes"),
                    )
                    remaining_target_indices = region_indices[
                        remaining_mask[region_indices]
                    ]
                                    
                    ### convert the assignment metal & remove the assignment
                    if len(remaining_target_indices) and metal["material_o"] in self.comps:
                        material_old = metal["material_o"]
                        material_new = metal["material"]
                        if material_new not in self.comps: 
                            self.comps[material_new] = len(self.comps)
                        comp_id_old = self.comps[material_old] 
                        comp_id_new = self.comps[material_new]
                        
                        assigned_mask = (
                            self.previous_element_2D_comp[
                                area_indices[remaining_target_indices]
                            ]
                            == comp_id_old
                        )
                        if np.any(assigned_mask):
                            remaining_assigned_indices = remaining_target_indices[assigned_mask]
                            self.element_2D_comp[area_indices[remaining_assigned_indices]] = comp_id_new
                            remaining_mask[remaining_assigned_indices] = False
            
            ### metal assignment Normal
            for metal in metals:
                if metal["type"] == "NORMAL": 
                    density = metal.get("density")
                    region_indices = self._search_faces(
                        area_indices,
                        metal.get("ranges"),
                        metal.get("holes"),
                    )
                    region_volumes = self.element_2D_volumn[
                        area_indices[region_indices]
                    ]
                    with np.errstate(over="ignore", invalid="ignore"):
                        volume = float(
                            np.sum(region_volumes, dtype=np.float64)
                        )
                    if not np.isfinite(volume):
                        raise OverflowError(
                            "NORMAL selector total area exceeds finite float64 "
                            "capacity"
                        )
                    # Preserve the existing externally visible field while
                    # consuming the selector immediately.
                    metal["volumn"] = volume
                    remaining_target_indices = region_indices[
                        remaining_mask[region_indices]
                    ]
                                                
                    ### assign metal
                    if len(remaining_target_indices) and density > 0:
                        ### find the assignment area
                        target_volumes = self.element_2D_volumn[area_indices[remaining_target_indices]]
                        remaining_assigned_indices = self._assign_metal(target_volumes, density, volume, randomSeed=layer)
                        assigned_indices  = remaining_target_indices[remaining_assigned_indices]
                        
                        ### assigne metal
                        material = metal["material"]
                        if material not in self.comps:
                            self.comps[material] = len(self.comps)
                        comp_id = self.comps[material]

                        ### assign the metal
                        self.element_2D_comp[area_indices[assigned_indices]] = comp_id

                        ### remove assigned element
                        remaining_mask[assigned_indices] = False

            ### assign the material
            remaining_indices = np.flatnonzero(remaining_mask).astype(
                np.int32,
                copy=False,
            )
            if len(remaining_indices):
                material = area["material"]
                if material not in self.comps:
                    self.comps[material] = len(self.comps)
                comp_id = self.comps[material]
                self.element_2D_comp[area_indices[remaining_indices]] = comp_id

        if inherited_full_footprint:
            self._element_2D_active_indices = previous_active
        elif active_chunks:
            if len(active_chunks) == 1:
                active_indices = active_chunks[0]
            else:
                active_indices = np.unique(np.concatenate(active_chunks))
            self._element_2D_active_indices = active_indices[
                self.element_2D_comp[active_indices] != self.comps["EMPTY"]
            ].astype(np.int32, copy=False)
        
    def _organize_empty(self):
        if len(self._element_2D_active_indices):
            self.element_2D_comp[self._element_2D_active_indices] = 0
        if len(self._previous_element_2D_active_indices):
            self.previous_element_2D_comp[
                self._previous_element_2D_active_indices
            ] = 0
        self._element_2D_active_indices = np.empty(0, dtype=np.int32)
        self._previous_element_2D_active_indices = np.empty(0, dtype=np.int32)
        self._clear_tracked_node_mapping()
        self._latest_top_hex_by_2D_element[:] = -1

    def _clear_tracked_node_mapping(self):
        """Clear only 2D nodes that currently map to a 3D top-plane node."""
        if len(self._mapped_node_2D_indices):
            self.node_2D_to_3D[self._mapped_node_2D_indices] = -1
        self._mapped_node_2D_indices = np.empty(0, dtype=np.int32)

    def validate_top_plane_xy_update(
        self,
        node_2D_ids,
        new_xy=None,
        incident_element_ids=None,
        chunk_size=_VALIDATION_CHUNK_SIZE,
    ):
        """Validate a prospective sparse update of the current 3D top plane.

        A generated hex linearly interpolates corresponding bottom and top
        quad corners.  Each corner turn over normalized height ``t`` is an
        exact quadratic.  Checking its endpoints and any interior minimum
        proves every cross-section remains strictly convex and CCW, without
        scanning previously generated 3D elements.

        Args:
            node_2D_ids: Touched 2D node ids.
            new_xy: Prospective coordinates aligned with ``node_2D_ids``.  If
                omitted, uses the current values in ``self.node_2D``.
            incident_element_ids: Optional complete set of touched 2D element
                ids supplied by an existing mesh adjacency index.  If omitted,
                Dragger lazily builds its own 2D node-to-element CSR once.
            chunk_size: Maximum number of local top hexes checked per chunk.

        Returns:
            Number of latest top hexahedra validated.
        """
        if not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        prepared = self._prepare_top_plane_xy_update(node_2D_ids, new_xy)
        node_2D_ids, node_3D_ids, targets = prepared
        if not len(node_3D_ids):
            return 0

        if incident_element_ids is None:
            incident_element_ids = self._incident_2D_element_ids(node_2D_ids)
        else:
            incident_element_ids = self._normalize_2D_element_ids(
                incident_element_ids
            )

        hex_ids = self._affected_latest_top_hex_ids(
            incident_element_ids,
            node_3D_ids,
        )
        self._validate_top_hex_transitions(
            hex_ids,
            node_3D_ids,
            targets,
            chunk_size,
        )
        return int(len(hex_ids))

    def sync_top_plane_xy(
        self,
        node_2D_ids,
        new_xy=None,
        incident_element_ids=None,
        chunk_size=_VALIDATION_CHUNK_SIZE,
    ):
        """Validate then atomically commit a sparse current-top XY update."""
        prepared = self._prepare_top_plane_xy_update(node_2D_ids, new_xy)
        normalized_2D_ids, node_3D_ids, targets = prepared
        if not len(node_3D_ids):
            return 0

        if incident_element_ids is None:
            incident_element_ids = self._incident_2D_element_ids(
                normalized_2D_ids
            )
        else:
            incident_element_ids = self._normalize_2D_element_ids(
                incident_element_ids
            )
        hex_ids = self._affected_latest_top_hex_ids(
            incident_element_ids,
            node_3D_ids,
        )
        self._validate_top_hex_transitions(
            hex_ids,
            node_3D_ids,
            targets,
            chunk_size,
        )

        # No mutation occurs until every affected transition has passed.
        self.nodes[node_3D_ids, :2] = targets
        return int(len(node_3D_ids))

    def _prepare_top_plane_xy_update(self, node_2D_ids, new_xy):
        node_2D_ids = np.asarray(node_2D_ids, dtype=np.intp)
        if node_2D_ids.ndim == 0:
            node_2D_ids = node_2D_ids.reshape(1)
        if node_2D_ids.ndim != 1:
            raise ValueError("node_2D_ids must be one-dimensional")
        if node_2D_ids.size and (
            int(node_2D_ids.min()) < 0
            or int(node_2D_ids.max()) >= len(self.node_2D)
        ):
            raise ValueError("node_2D_ids contain out-of-range ids")

        if new_xy is None:
            targets = self.node_2D[node_2D_ids, :2].copy()
        else:
            targets = np.asarray(new_xy, dtype=np.float64)
            if targets.ndim == 1 and len(node_2D_ids) == 1 and targets.shape == (2,):
                targets = targets.reshape(1, 2)
            if targets.shape != (len(node_2D_ids), 2):
                raise ValueError("new_xy must have shape (len(node_2D_ids), 2)")
            targets = targets.copy()
        if not np.all(np.isfinite(targets)):
            raise ValueError("new_xy must contain finite coordinates")

        # Normalize duplicate API inputs while rejecting conflicting targets.
        if len(node_2D_ids):
            order = np.argsort(node_2D_ids, kind="stable")
            node_2D_ids = node_2D_ids[order]
            targets = targets[order]
            duplicate = node_2D_ids[1:] == node_2D_ids[:-1]
            if np.any(
                duplicate
                & np.any(targets[1:] != targets[:-1], axis=1)
            ):
                raise ValueError("one 2D node received conflicting XY targets")
            keep = np.concatenate(([True], ~duplicate))
            node_2D_ids = node_2D_ids[keep]
            targets = targets[keep]

        node_3D_ids = self.node_2D_to_3D[node_2D_ids]
        valid = node_3D_ids >= 0
        node_2D_ids = node_2D_ids[valid]
        node_3D_ids = node_3D_ids[valid].astype(np.intp, copy=False)
        targets = targets[valid]
        if not len(node_3D_ids):
            return node_2D_ids, node_3D_ids, targets

        order = np.argsort(node_3D_ids, kind="stable")
        node_2D_ids = node_2D_ids[order]
        node_3D_ids = node_3D_ids[order]
        targets = targets[order]
        if np.any(node_3D_ids[1:] == node_3D_ids[:-1]):
            raise ValueError("multiple 2D nodes alias the same current top node")
        return node_2D_ids, node_3D_ids, targets

    def _normalize_2D_element_ids(self, element_ids):
        element_ids = np.asarray(element_ids, dtype=np.intp)
        if element_ids.ndim == 0:
            element_ids = element_ids.reshape(1)
        if element_ids.ndim != 1:
            raise ValueError("incident_element_ids must be one-dimensional")
        element_ids = np.unique(element_ids)
        if element_ids.size and (
            int(element_ids.min()) < 0
            or int(element_ids.max()) >= len(self.element_2D)
        ):
            raise ValueError("incident_element_ids contain out-of-range ids")
        return element_ids

    def _incident_2D_element_ids(self, node_2D_ids):
        if self._node_2D_element_offsets is None:
            self._build_2D_node_element_csr()
        node_2D_ids = np.unique(np.asarray(node_2D_ids, dtype=np.intp))
        if not len(node_2D_ids):
            return np.empty(0, dtype=np.intp)

        offsets = self._node_2D_element_offsets
        lengths = offsets[node_2D_ids + 1] - offsets[node_2D_ids]
        total = int(lengths.sum())
        if total == 0:
            return np.empty(0, dtype=np.intp)
        group_starts = np.cumsum(lengths) - lengths
        positions = (
            np.repeat(offsets[node_2D_ids], lengths)
            + np.arange(total, dtype=np.intp)
            - np.repeat(group_starts, lengths)
        )
        return np.unique(self._node_2D_element_ids[positions]).astype(
            np.intp,
            copy=False,
        )

    def _build_2D_node_element_csr(self):
        node_count = len(self.node_2D)
        if not len(self.element_2D):
            self._node_2D_element_offsets = np.zeros(
                node_count + 1,
                dtype=np.intp,
            )
            self._node_2D_element_ids = np.empty(0, dtype=np.int32)
            return

        flat_node_ids = self.element_2D.ravel()
        counts = np.bincount(flat_node_ids, minlength=node_count)
        offsets = np.empty(node_count + 1, dtype=np.intp)
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
        order = np.argsort(flat_node_ids, kind="stable")
        element_dtype = np.int32 if len(self.element_2D) < 2**31 else np.int64
        flat_element_ids = np.repeat(
            np.arange(len(self.element_2D), dtype=element_dtype),
            4,
        )
        self._node_2D_element_offsets = offsets
        self._node_2D_element_ids = flat_element_ids[order]

    def _affected_latest_top_hex_ids(self, incident_element_ids, node_3D_ids):
        if not len(incident_element_ids):
            return np.empty(0, dtype=np.intp)
        hex_ids = self._latest_top_hex_by_2D_element[incident_element_ids]
        hex_ids = np.unique(hex_ids[hex_ids >= 0]).astype(np.intp, copy=False)
        if not len(hex_ids):
            return hex_ids
        if int(hex_ids.max()) >= self.element_num:
            raise RuntimeError("latest top hex index is inconsistent")

        top_node_ids = self.elements[hex_ids, 4:]
        positions = np.searchsorted(node_3D_ids, top_node_ids)
        safe_positions = np.minimum(positions, len(node_3D_ids) - 1)
        affected = (positions < len(node_3D_ids)) & (
            node_3D_ids[safe_positions] == top_node_ids
        )
        return hex_ids[affected.any(axis=1)]

    def _validate_top_hex_transitions(
        self,
        hex_ids,
        node_3D_ids,
        targets,
        chunk_size,
    ):
        for start in range(0, len(hex_ids), chunk_size):
            chunk_hex_ids = hex_ids[start : start + chunk_size]
            connectivity = self.elements[chunk_hex_ids]
            bottom_points = self.nodes[connectivity[:, :4]]
            top_points = self.nodes[connectivity[:, 4:]].copy()

            bottom_z = bottom_points[:, :, 2]
            top_z = top_points[:, :, 2]
            planar = (
                np.all(bottom_z == bottom_z[:, :1], axis=1)
                & np.all(top_z == top_z[:, :1], axis=1)
                & (top_z[:, 0] > bottom_z[:, 0])
            )
            if not np.all(planar):
                invalid = int(np.flatnonzero(~planar)[0])
                raise ValueError(
                    f"3D element {int(chunk_hex_ids[invalid])} does not have "
                    "strictly ordered planar z faces"
                )

            flat_top_ids = connectivity[:, 4:].reshape(-1)
            positions = np.searchsorted(node_3D_ids, flat_top_ids)
            safe_positions = np.minimum(positions, len(node_3D_ids) - 1)
            changed = (positions < len(node_3D_ids)) & (
                node_3D_ids[safe_positions] == flat_top_ids
            )
            flat_top_xy = top_points[:, :, :2].reshape(-1, 2)
            flat_top_xy[changed] = targets[safe_positions[changed]]

            bottom_xy = bottom_points[:, :, :2]
            top_xy = top_points[:, :, :2]
            if not np.all(np.isfinite(bottom_xy)) or not np.all(np.isfinite(top_xy)):
                raise ValueError("3D transition contains non-finite XY coordinates")

            bottom_edges = np.roll(bottom_xy, -1, axis=1) - bottom_xy
            top_edges = np.roll(top_xy, -1, axis=1) - top_xy
            edge_delta = top_edges - bottom_edges
            next_bottom = np.roll(bottom_edges, -1, axis=1)
            next_delta = np.roll(edge_delta, -1, axis=1)

            coefficient_c = (
                bottom_edges[:, :, 0] * next_bottom[:, :, 1]
                - bottom_edges[:, :, 1] * next_bottom[:, :, 0]
            )
            coefficient_b = (
                edge_delta[:, :, 0] * next_bottom[:, :, 1]
                - edge_delta[:, :, 1] * next_bottom[:, :, 0]
                + bottom_edges[:, :, 0] * next_delta[:, :, 1]
                - bottom_edges[:, :, 1] * next_delta[:, :, 0]
            )
            coefficient_a = (
                edge_delta[:, :, 0] * next_delta[:, :, 1]
                - edge_delta[:, :, 1] * next_delta[:, :, 0]
            )

            endpoint_one = coefficient_a + coefficient_b + coefficient_c
            minimum = np.minimum(coefficient_c, endpoint_one)
            convex_quadratic = coefficient_a > 0.0
            vertex_t = np.zeros_like(coefficient_a)
            vertex_t[convex_quadratic] = (
                -coefficient_b[convex_quadratic]
                / (2.0 * coefficient_a[convex_quadratic])
            )
            interior_vertex = convex_quadratic & (vertex_t > 0.0) & (vertex_t < 1.0)
            vertex_value = (
                (coefficient_a * vertex_t + coefficient_b) * vertex_t
                + coefficient_c
            )
            minimum = np.where(
                interior_vertex,
                np.minimum(minimum, vertex_value),
                minimum,
            )

            invalid = ~np.isfinite(minimum) | (minimum <= 0.0)
            if np.any(invalid):
                local_hex, corner = np.argwhere(invalid)[0]
                raise ValueError(
                    "Top-plane update would create a degenerate or inverted "
                    f"transition in 3D element {int(chunk_hex_ids[local_hex])} "
                    f"at corner {int(corner)}"
                )
        
    def _drag(self, element_size: float, begin: float, end: float):
        begin = float(begin)
        end = float(end)
        element_size = float(element_size)
        if not np.isfinite(begin) or not np.isfinite(end):
            raise ValueError("drag begin and end must be finite")
        if end <= begin:
            raise ValueError("drag interval must satisfy begin < end")
        if not np.isfinite(element_size) or element_size <= 0.0:
            raise ValueError("drag element_size must be finite and greater than zero")

        # element_size is a maximum, not merely a preference.  Do not round
        # the distance: thin but positive layers must still produce one hex.
        distance = end - begin
        subdivision_ratio = distance / element_size
        if not np.isfinite(distance) or not np.isfinite(subdivision_ratio):
            raise OverflowError("drag interval requires too many subdivisions")
        drag_num = max(1, int(np.ceil(subdivision_ratio)))
        actual_element_size = distance / drag_num

        ### target element index
        elem2D_idx = np.asarray(
            self._element_2D_active_indices,
            dtype=np.intp,
        )
        if elem2D_idx.size == 0:
            self._clear_tracked_node_mapping()
            return 0
        if np.any(self.element_2D_comp[elem2D_idx] == self.comps["EMPTY"]):
            raise RuntimeError("active 2D element index points to EMPTY material")

        # unique 2D node ids used by those elements
        elem2D_nodes = self.element_2D[elem2D_idx]
        node2D_idx, inv = np.unique(elem2D_nodes, return_inverse=True)
        elem2D_nodes_local = inv.reshape(elem2D_nodes.shape).astype(
            np.int32,
            copy=False,
        )

        N = int(node2D_idx.size)
        E = int(elem2D_idx.size)
        new_node_count = N * drag_num
        new_element_count = E * drag_num
        int32_max = np.iinfo(np.int32).max
        if self.element_num + new_element_count > int32_max:
            raise OverflowError("3D element count exceeds int32 id capacity")

        # Prove the exact float64 planes before changing node mappings or
        # allocating output.  At large absolute z, a requested subdivision can
        # round two adjacent planes to the same number and create zero-volume
        # hexahedra even though begin < end in real arithmetic.
        self._validate_drag_z_planes(
            begin,
            end,
            element_size,
            drag_num,
            actual_element_size,
        )

        ### add the unexisted node
        unexisted_node2D_idx = node2D_idx[self.node_2D_to_3D[node2D_idx] == -1]
        if (
            self.node_num
            + int(unexisted_node2D_idx.size)
            + new_node_count
            > int32_max
        ):
            raise OverflowError("3D node count exceeds int32 connectivity capacity")
        if unexisted_node2D_idx.size:
            self._pre_allocate_nodes(unexisted_node2D_idx.size)

            initial_start = self.node_num
            initial_end = initial_start + int(unexisted_node2D_idx.size)
            dst = self.nodes[initial_start:initial_end]
            np.take(self.node_2D, unexisted_node2D_idx, axis=0, out=dst[:, :2]) # xy from 2D nodes -> write directly without temp:
            dst[:, 2] = begin

            initial_ids = np.arange(initial_start, initial_end, dtype=np.int32)
            self.node_2D_to_3D[unexisted_node2D_idx] = initial_ids
            self.node_ids[initial_start:initial_end] = initial_ids + 1
            self.node_num = initial_end

        ### begin to drag
        base_map = self.node_2D_to_3D[node2D_idx]

        ### [NODE] allocate final storage, then populate it in plane chunks.
        self._pre_allocate_nodes(new_node_count)
        node_start = self.node_num
        xy = self.node_2D[node2D_idx]
        planes_per_chunk = max(1, _CONNECTIVITY_CHUNK_SIZE // max(N, 1))
        for plane_start in range(0, drag_num, planes_per_chunk):
            plane_end = min(plane_start + planes_per_chunk, drag_num)
            count = plane_end - plane_start
            dst_start = node_start + plane_start * N
            dst_end = node_start + plane_end * N
            dst_nodes = self.nodes[dst_start:dst_end].reshape(count, N, 3)
            dst_nodes[:, :, :2] = xy

            plane_numbers = np.arange(
                plane_start + 1,
                plane_end + 1,
                dtype=np.float64,
            )
            z_values = begin + actual_element_size * plane_numbers
            if plane_end == drag_num:
                z_values[-1] = end
            dst_nodes[:, :, 2] = z_values[:, None]

            self.node_ids[dst_start:dst_end] = np.arange(
                dst_start + 1,
                dst_end + 1,
                dtype=np.int32,
            )

        self.node_num += new_node_count

        ### [ELEMENT] write connectivity directly into final storage.  Chunk
        # by the flat final-hexahedron index so Python work scales with the
        # number of bounded chunks, not with the number of z planes.
        self._pre_allocate_elements(new_element_count)
        elem_start = self.element_num
        layer_comps = self.element_2D_comp[elem2D_idx]
        for flat_start in range(
            0,
            new_element_count,
            _CONNECTIVITY_CHUNK_SIZE,
        ):
            flat_end = min(
                flat_start + _CONNECTIVITY_CHUNK_SIZE,
                new_element_count,
            )
            flat_indices = np.arange(flat_start, flat_end, dtype=np.int64)
            plane_indices = flat_indices // E
            local_indices = (flat_indices - plane_indices * E).astype(
                np.intp,
                copy=False,
            )
            local_connectivity = elem2D_nodes_local[local_indices]
            out_start = elem_start + flat_start
            out_end = elem_start + flat_end
            destination = self.elements[out_start:out_end]

            _write_drag_connectivity_chunk(
                destination,
                local_connectivity,
                plane_indices,
                base_map,
                node_start,
                N,
            )
            self.element_ids[out_start:out_end] = np.arange(
                out_start + 1,
                out_end + 1,
                dtype=np.int32,
            )
            self.element_comps[out_start:out_end] = layer_comps[
                local_indices
            ]

        self.element_num += new_element_count

        latest_hex_start = elem_start + (drag_num - 1) * E
        self._latest_top_hex_by_2D_element[elem2D_idx] = (
            latest_hex_start + np.arange(E, dtype=np.int32)
        )

        # update map so future ops start from the top layer
        self._clear_tracked_node_mapping()
        last_plane_start = node_start + (drag_num - 1) * N
        self.node_2D_to_3D[node2D_idx] = last_plane_start + np.arange(
            N,
            dtype=np.int32,
        )
        self._mapped_node_2D_indices = node2D_idx.astype(
            np.int32,
            copy=False,
        )
        return drag_num

    @staticmethod
    def _validate_drag_z_planes(
        begin,
        end,
        element_size,
        drag_num,
        actual_element_size,
        chunk_size=_VALIDATION_CHUNK_SIZE,
    ):
        """Require strictly increasing representable z planes in chunks."""
        if not np.isfinite(actual_element_size) or actual_element_size <= 0.0:
            raise ValueError(
                "drag z subdivision is not representable in float64"
            )
        forward_resolution = float(np.nextafter(begin, end) - begin)
        backward_resolution = float(end - np.nextafter(end, begin))
        representable_resolution = max(
            forward_resolution,
            backward_resolution,
        )
        if representable_resolution > element_size:
            raise ValueError(
                "drag element_size is smaller than the float64 z resolution"
            )
        # Arithmetic plane differences can exceed the mathematical spacing by
        # a few representable steps.  This allowance affects sizing only;
        # topology still requires every plane to be strictly increasing.
        spacing_limit = element_size + 2.0 * representable_resolution
        previous = begin
        for plane_start in range(0, drag_num, chunk_size):
            plane_end = min(plane_start + chunk_size, drag_num)
            plane_numbers = np.arange(
                plane_start + 1,
                plane_end + 1,
                dtype=np.float64,
            )
            z_values = begin + actual_element_size * plane_numbers
            if plane_end == drag_num:
                z_values[-1] = end
            if (
                not np.all(np.isfinite(z_values))
                or z_values[0] <= previous
                or np.any(np.diff(z_values) <= 0.0)
            ):
                raise ValueError(
                    "drag z subdivision would create duplicate or reversed "
                    "float64 planes"
                )
            interval_sizes = np.diff(
                np.concatenate((np.asarray([previous]), z_values))
            )
            if np.any(interval_sizes > spacing_limit):
                raise ValueError(
                    "drag z subdivision cannot satisfy element_size in "
                    "float64"
                )
            previous = float(z_values[-1])
        if previous != end:
            raise ValueError("drag z subdivision does not reach the exact end")

    def build(self, object_list):
        for index, obj in enumerate(object_list):
            self._organize_empty()
            for index, layer in enumerate(obj[:-1]):
                self._organize(layer["areas"], index)
                self._drag(layer["element_size"], obj[index]["z"], obj[index+1]["z"])
                
    def _cal_volumns(self, element_indices=None, chunk_size=_VALIDATION_CHUNK_SIZE):
        """Validate CCW convex quads and update their signed areas.

        Validation and area calculation share the same chunked pass.  This
        deliberately never applies ``abs``: clockwise, collapsed, concave, or
        self-intersecting quadrilaterals are unsafe FEM cells and fail closed.
        ``element_indices`` allows callers with node-to-element adjacency to
        update only quads touched by a sparse snap operation.
        """
        if not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")

        if element_indices is None:
            total = len(self.element_2D)
            chunks = (
                (start, min(start + chunk_size, total), None)
                for start in range(0, total, chunk_size)
            )
        else:
            normalized = np.asarray(element_indices, dtype=np.intp)
            if normalized.ndim == 0:
                normalized = normalized.reshape(1)
            if normalized.ndim != 1:
                raise ValueError("element_indices must be one-dimensional")
            if normalized.size and (
                int(normalized.min()) < 0
                or int(normalized.max()) >= len(self.element_2D)
            ):
                raise ValueError("element_indices contain out-of-range ids")
            chunks = (
                (
                    start,
                    min(start + chunk_size, len(normalized)),
                    normalized[start : min(start + chunk_size, len(normalized))],
                )
                for start in range(0, len(normalized), chunk_size)
            )

        for start, end, selected_indices in chunks:
            if selected_indices is None:
                connectivity = self.element_2D[start:end]
            else:
                connectivity = self.element_2D[selected_indices]
            if not len(connectivity):
                continue

            corner_xy = self.node_2D[connectivity]
            if not np.all(np.isfinite(corner_xy)):
                local_invalid = int(
                    np.flatnonzero(~np.isfinite(corner_xy).all(axis=(1, 2)))[0]
                )
                element_id = (
                    start + local_invalid
                    if selected_indices is None
                    else int(selected_indices[local_invalid])
                )
                raise ValueError(
                    f"2D element {element_id} contains non-finite coordinates"
                )

            # Shift each cell to its first corner before cross products.  This
            # avoids cancellation when a small cell has a large global offset.
            shifted = corner_xy - corner_xy[:, :1, :]
            edge0 = shifted[:, 1] - shifted[:, 0]
            edge1 = shifted[:, 2] - shifted[:, 1]
            edge2 = shifted[:, 3] - shifted[:, 2]
            edge3 = shifted[:, 0] - shifted[:, 3]

            cross01 = _cross_2d(edge0, edge1)
            cross12 = _cross_2d(edge1, edge2)
            cross23 = _cross_2d(edge2, edge3)
            cross30 = _cross_2d(edge3, edge0)
            area2 = (
                _cross_2d(shifted[:, 1], shifted[:, 2])
                + _cross_2d(shifted[:, 2], shifted[:, 3])
            )

            invalid_area = ~np.isfinite(area2) | (area2 <= 0.0)
            if np.any(invalid_area):
                local_invalid = int(np.flatnonzero(invalid_area)[0])
                element_id = (
                    start + local_invalid
                    if selected_indices is None
                    else int(selected_indices[local_invalid])
                )
                raise ValueError(
                    "2D element "
                    f"{element_id} has non-positive signed area; "
                    "quadrilaterals must be non-degenerate and counter-clockwise"
                )

            invalid_corner = (
                (cross01 <= 0.0)
                | (cross12 <= 0.0)
                | (cross23 <= 0.0)
                | (cross30 <= 0.0)
            )
            if np.any(invalid_corner):
                local_invalid = int(np.flatnonzero(invalid_corner)[0])
                element_id = (
                    start + local_invalid
                    if selected_indices is None
                    else int(selected_indices[local_invalid])
                )
                raise ValueError(
                    "2D element "
                    f"{element_id} is concave, self-intersecting, or has a "
                    "degenerate corner"
                )

            areas = 0.5 * area2
            if selected_indices is None:
                self.element_2D_volumn[start:end] = areas
            else:
                self.element_2D_volumn[selected_indices] = areas


Engin25D = Dragger
