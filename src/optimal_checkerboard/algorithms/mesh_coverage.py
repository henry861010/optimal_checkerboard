"""Topology coverage checks for user-assigned checkerboard meshes.

The checks in this module intentionally validate mesh *edges*, rather than
only looking for nodes at feature-span endpoints.  A custom mesh is safe to
snap only when every feature has a connected, collinear edge chain on its
structural shared rail.

Mandatory axis coordinates have a related but slightly different meaning. A
mandatory x coordinate is a complete vertical station (and a mandatory y
coordinate a complete horizontal station): no active quadrilateral may cross
the coordinate through its interior, and the coordinate must own at least one
positive-length mesh edge.  Disconnected footprint sections are allowed; the
station is complete independently in every section it intersects.

Element connectivity is scanned in chunks.  Only edges lying on coordinates
needed by a rail rule or mandatory station are retained, so memory use scales
with the relevant grid lines instead of all mesh edges.
"""

from dataclasses import dataclass

import numpy as np


_AXIS_DATA = {
    "x": (0, 1),
    "y": (1, 0),
}


@dataclass(frozen=True)
class _StationTopology:
    """Compact one-dimensional topology for a structural mesh station."""

    node_ids: np.ndarray
    span_values: np.ndarray
    component_ids: np.ndarray


def validate_mesh_coverage(
    nodes,
    elements,
    rails,
    snap_rules,
    mandatory_coordinates=None,
    *,
    eps=0.0,
    chunk_size=250_000,
):
    """Validate feature edge chains and mandatory axis stations.

    Args:
        nodes: ``(n, 2+)`` floating-point mesh node coordinates.
        elements: ``(m, 4)`` integer quadrilateral connectivity.
        rails: Mapping with ``"x"`` and ``"y"`` rail lists.  Each rail must
            provide its structural ``coord``; list position is its rail id.
        snap_rules: Either a z-indexed mapping of rule lists or a flat rule
            iterable.  Every rule's ``rail_id`` and feature span are checked.
        mandatory_coordinates: Optional ``{"x": [...], "y": [...]}``
            mapping.  A complete station cannot pass through the interior of
            an active element and must contain a positive-length mesh edge.
        eps: Explicit coordinate tolerance.  The default is exact comparison;
            a positive value is accepted only when requested by the caller.
        chunk_size: Maximum quadrilaterals inspected in one vectorized batch.

    Returns:
        A small dictionary containing validation counts.  No mesh-sized
        output is retained after this function returns.

    Raises:
        ValueError: If arrays, rail metadata, feature coverage, or a mandatory
            station is invalid.
    """
    nodes, elements = _validate_mesh_arrays(nodes, elements)
    eps = _non_negative_finite_number(eps, "eps")
    chunk_size = _positive_integer(chunk_size, "chunk_size")
    rails_by_axis = _normalize_rails(rails)
    rules = _normalize_snap_rules(snap_rules, rails_by_axis, eps)
    mandatory = _normalize_axis_coordinates(
        mandatory_coordinates,
        "mandatory_coordinates",
    )

    required_coordinates = {"x": set(), "y": set()}
    for rule in rules:
        required_coordinates[rule["axis"]].add(rule["rail_coord"])
    for axis in ("x", "y"):
        required_coordinates[axis].update(mandatory[axis])

    station_coordinates = {
        axis: np.asarray(sorted(values), dtype=np.float64)
        for axis, values in required_coordinates.items()
    }
    _validate_coordinate_separation(station_coordinates, eps)

    retained_edges = _collect_station_edges(
        nodes,
        elements,
        station_coordinates,
        eps=eps,
        chunk_size=chunk_size,
    )
    station_topologies = {
        axis: _build_axis_station_topologies(
            nodes,
            retained_edges[axis],
            station_coordinates[axis],
            axis,
            eps,
        )
        for axis in ("x", "y")
    }

    _validate_feature_edge_chains(
        rules,
        station_coordinates,
        station_topologies,
        eps,
    )
    _validate_mandatory_stations(
        nodes,
        elements,
        mandatory,
        station_coordinates,
        station_topologies,
        eps=eps,
        chunk_size=chunk_size,
    )

    return {
        "rule_count": len(rules),
        "mandatory_station_count": sum(len(values) for values in mandatory.values()),
        "retained_edge_count": sum(
            len(edge_array)
            for edge_array in retained_edges.values()
        ),
    }


def validate_box_domain_partition(
    nodes,
    elements,
    bounds,
    *,
    chunk_size=250_000,
):
    """Prove that a general quadrilateral mesh partitions one BOX domain.

    The proof is intentionally stronger than checking feature rails: every
    referenced node must be inside the BOX, every undirected edge must have
    manifold multiplicity one or two, all multiplicity-one edges must lie on
    the physical BOX boundary, and those edges must continuously cover all
    four sides.  A high-precision area checksum additionally rejects crossing
    or overlapping embeddings that edge incidence alone cannot identify.

    This general-mesh check retains four integer edge keys per element.  Very
    large row-major grids should provide verified ``STRUCTURED_BOX`` metadata,
    whose topology proof uses compact axis arithmetic instead.
    """
    nodes, elements = _validate_mesh_arrays(nodes, elements)
    chunk_size = _positive_integer(chunk_size, "chunk_size")
    try:
        xmin, ymin, xmax, ymax = [float(value) for value in bounds]
    except (TypeError, ValueError) as exc:
        raise ValueError("BOX bounds must contain four finite numbers") from exc
    if not np.all(np.isfinite((xmin, ymin, xmax, ymax))):
        raise ValueError("BOX bounds must contain four finite numbers")
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("BOX bounds must have positive area")
    if not len(elements):
        raise ValueError("custom mesh does not cover the BOX domain")
    if len(nodes) >= 2**32:
        raise OverflowError("custom mesh node ids exceed partition key capacity")

    edge_key_chunks = []
    edge_direction_chunks = []
    area_total = np.longdouble(0.0)
    edge_offsets = ((0, 1), (1, 2), (2, 3), (3, 0))
    for start in range(0, len(elements), chunk_size):
        chunk = elements[start : start + chunk_size]
        chunk_node_ids = chunk.ravel()
        xy = nodes[chunk_node_ids, :2]
        if (
            np.any(xy[:, 0] < xmin)
            or np.any(xy[:, 0] > xmax)
            or np.any(xy[:, 1] < ymin)
            or np.any(xy[:, 1] > ymax)
        ):
            raise ValueError(
                "custom mesh contains active nodes outside the BOX domain"
            )

        points = nodes[chunk, :2].astype(np.longdouble, copy=False)
        shifted = points - points[:, :1, :]
        area2 = (
            shifted[:, 1, 0] * shifted[:, 2, 1]
            - shifted[:, 1, 1] * shifted[:, 2, 0]
            + shifted[:, 2, 0] * shifted[:, 3, 1]
            - shifted[:, 2, 1] * shifted[:, 3, 0]
        )
        if np.any(~np.isfinite(area2)) or np.any(area2 <= 0.0):
            raise ValueError(
                "custom BOX mesh contains non-positive quadrilateral area"
            )
        area_total += np.sum(area2, dtype=np.longdouble) * np.longdouble(0.5)

        for first, second in edge_offsets:
            edge_u = chunk[:, first].astype(np.uint64, copy=False)
            edge_v = chunk[:, second].astype(np.uint64, copy=False)
            low = np.minimum(edge_u, edge_v)
            high = np.maximum(edge_u, edge_v)
            edge_key_chunks.append((low << np.uint64(32)) | high)
            edge_direction_chunks.append(
                np.where(edge_u == low, 1, -1).astype(np.int8)
            )

    edge_keys = np.concatenate(edge_key_chunks)
    edge_directions = np.concatenate(edge_direction_chunks)
    unique_keys, inverse_edges, edge_counts = np.unique(
        edge_keys,
        return_inverse=True,
        return_counts=True,
    )
    if np.any((edge_counts != 1) & (edge_counts != 2)):
        raise ValueError(
            "custom BOX mesh has duplicate elements or non-manifold edges"
        )
    orientation_balance = np.bincount(
        inverse_edges,
        weights=edge_directions,
        minlength=len(unique_keys),
    )
    if np.any(orientation_balance[edge_counts == 2] != 0.0):
        raise ValueError(
            "custom BOX mesh has duplicate elements or inconsistently "
            "oriented interior edges"
        )
    edge_key_chunks.clear()
    edge_direction_chunks.clear()
    del edge_keys, edge_directions, inverse_edges, orientation_balance

    boundary_keys = unique_keys[edge_counts == 1]
    if not len(boundary_keys):
        raise ValueError("custom BOX mesh has no physical domain boundary")
    boundary_u = (boundary_keys >> np.uint64(32)).astype(np.intp)
    boundary_v = (boundary_keys & np.uint64(0xFFFFFFFF)).astype(np.intp)
    point_u = nodes[boundary_u, :2]
    point_v = nodes[boundary_v, :2]

    side_intervals = {"left": [], "right": [], "bottom": [], "top": []}
    side_masks = {
        "left": (point_u[:, 0] == xmin) & (point_v[:, 0] == xmin),
        "right": (point_u[:, 0] == xmax) & (point_v[:, 0] == xmax),
        "bottom": (point_u[:, 1] == ymin) & (point_v[:, 1] == ymin),
        "top": (point_u[:, 1] == ymax) & (point_v[:, 1] == ymax),
    }
    on_domain_boundary = np.zeros(len(boundary_keys), dtype=bool)
    for side, mask in side_masks.items():
        on_domain_boundary |= mask
        span_axis = 1 if side in ("left", "right") else 0
        side_intervals[side].extend(
            (
                min(float(first), float(second)),
                max(float(first), float(second)),
            )
            for first, second in zip(
                point_u[mask, span_axis],
                point_v[mask, span_axis],
            )
        )
    if not np.all(on_domain_boundary):
        raise ValueError(
            "custom BOX mesh has a hole, disconnected region, or internal "
            "unmatched edge"
        )

    for side in ("left", "right"):
        if not _closed_interval_union_covers(
            side_intervals[side],
            ymin,
            ymax,
        ):
            raise ValueError(f"custom BOX mesh does not cover the {side} side")
    for side in ("bottom", "top"):
        if not _closed_interval_union_covers(
            side_intervals[side],
            xmin,
            xmax,
        ):
            raise ValueError(f"custom BOX mesh does not cover the {side} side")

    expected_area = (
        (np.longdouble(xmax) - np.longdouble(xmin))
        * (np.longdouble(ymax) - np.longdouble(ymin))
    )
    area_scale = max(abs(expected_area), abs(area_total), np.longdouble(1.0))
    area_tolerance = (
        np.finfo(np.longdouble).eps
        * area_scale
        * np.longdouble(max(32, 8 * len(elements)))
    )
    if abs(area_total - expected_area) > area_tolerance:
        raise ValueError(
            "custom BOX mesh area does not exactly partition the domain"
        )

    return {
        "element_count": len(elements),
        "unique_edge_count": len(unique_keys),
        "boundary_edge_count": len(boundary_keys),
    }


def _closed_interval_union_covers(intervals, required_min, required_max):
    """Return whether exact closed intervals cover one complete span."""
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


def _validate_mesh_arrays(nodes, elements):
    try:
        nodes = np.asarray(nodes)
        elements = np.asarray(elements)
    except (TypeError, ValueError) as exc:
        raise ValueError("nodes and elements must be numeric arrays") from exc

    if nodes.ndim != 2 or nodes.shape[1] < 2:
        raise ValueError("nodes must have shape (n, 2+)")
    if not np.issubdtype(nodes.dtype, np.floating):
        raise ValueError("nodes must contain floating point coordinates")
    if not np.all(np.isfinite(nodes[:, :2])):
        raise ValueError("node x/y coordinates must be finite")
    if elements.ndim != 2 or elements.shape[1] != 4:
        raise ValueError("elements must have shape (m, 4)")
    if not np.issubdtype(elements.dtype, np.integer):
        raise ValueError("elements must contain integer node ids")
    if elements.size:
        if np.any(elements < 0) or int(elements.max()) >= len(nodes):
            raise ValueError("elements contain node ids outside nodes")
        if np.any(
            (elements[:, 0] == elements[:, 1])
            | (elements[:, 1] == elements[:, 2])
            | (elements[:, 2] == elements[:, 3])
            | (elements[:, 3] == elements[:, 0])
        ):
            raise ValueError("quadrilateral boundary edges must have two nodes")
    return nodes, elements


def _normalize_rails(rails):
    if not isinstance(rails, dict):
        raise ValueError("rails must be a dictionary with x/y lists")
    if set(rails) - {"x", "y"}:
        raise ValueError("rails contains unsupported axes")

    normalized = {"x": [], "y": []}
    for axis in ("x", "y"):
        seen_coords = set()
        for rail_id, rail in enumerate(rails.get(axis, ())):
            if not isinstance(rail, dict) or "coord" not in rail:
                raise ValueError(f"{axis}-axis rail {rail_id} has no coordinate")
            coord = _finite_number(
                rail["coord"],
                f"{axis}-axis rail {rail_id} coordinate",
            )
            if coord in seen_coords:
                raise ValueError(
                    f"{axis}-axis structural rails cannot share coordinate {coord}"
                )
            seen_coords.add(coord)
            declared_axis = rail.get("axis", axis)
            if declared_axis != axis:
                raise ValueError(f"{axis}-axis rail {rail_id} declares another axis")
            declared_id = rail.get("rail_id", rail_id)
            if declared_id != rail_id:
                raise ValueError(
                    f"{axis}-axis rail ids must match their list positions"
                )
            normalized[axis].append(coord)
    return normalized


def _normalize_snap_rules(snap_rules, rails_by_axis, eps):
    if snap_rules is None:
        rule_values = []
    elif isinstance(snap_rules, dict):
        rule_values = [
            rule
            for rules_at_z in snap_rules.values()
            for rule in rules_at_z
        ]
    else:
        try:
            rule_values = list(snap_rules)
        except TypeError as exc:
            raise ValueError("snap_rules must be a mapping or iterable") from exc

    normalized = []
    for index, rule in enumerate(rule_values):
        if not isinstance(rule, dict):
            raise ValueError(f"snap rule {index} must be a dictionary")
        try:
            axis = rule["axis"]
            rail_id = rule["rail_id"]
            span_min = _finite_number(rule["span_min"], "snap rule span_min")
            span_max = _finite_number(rule["span_max"], "snap rule span_max")
        except KeyError as exc:
            raise ValueError(f"snap rule {index} is missing {exc.args[0]}") from exc
        if axis not in _AXIS_DATA:
            raise ValueError(f"snap rule {index} has an unsupported axis")
        if isinstance(rail_id, bool) or not isinstance(rail_id, (int, np.integer)):
            raise ValueError(f"snap rule {index} rail_id must be an integer")
        rail_id = int(rail_id)
        if rail_id < 0 or rail_id >= len(rails_by_axis[axis]):
            raise ValueError(f"snap rule {index} references an unknown rail")
        if span_max - span_min <= 2.0 * eps:
            raise ValueError(f"snap rule {index} must have a positive span")

        rail_coord = rails_by_axis[axis][rail_id]
        if "rail_coord" in rule:
            declared_coord = _finite_number(
                rule["rail_coord"],
                "snap rule rail_coord",
            )
            if abs(declared_coord - rail_coord) > eps:
                raise ValueError(
                    f"snap rule {index} rail coordinate disagrees with rail metadata"
                )
        normalized.append(
            {
                "source_index": index,
                "axis": axis,
                "rail_id": rail_id,
                "rail_coord": rail_coord,
                "span_min": span_min,
                "span_max": span_max,
            }
        )
    return normalized


def _collect_station_edges(
    nodes,
    elements,
    station_coordinates,
    *,
    eps,
    chunk_size,
):
    """Retain only collinear quad edges on coordinates being validated."""
    edge_chunks = {"x": [], "y": []}
    edge_offsets = ((0, 1), (1, 2), (2, 3), (3, 0))

    for start in range(0, len(elements), chunk_size):
        chunk = elements[start : start + chunk_size]
        if not len(chunk):
            continue
        edge_u = np.concatenate([chunk[:, first] for first, _ in edge_offsets])
        edge_v = np.concatenate([chunk[:, second] for _, second in edge_offsets])

        for axis, (coord_axis, span_axis) in _AXIS_DATA.items():
            coordinates = station_coordinates[axis]
            if not len(coordinates):
                continue
            coord_u = nodes[edge_u, coord_axis]
            coord_v = nodes[edge_v, coord_axis]
            span_u = nodes[edge_u, span_axis]
            span_v = nodes[edge_v, span_axis]
            collinear = (
                (np.abs(coord_u - coord_v) <= eps)
                & (np.abs(span_u - span_v) > eps)
            )
            if not np.any(collinear):
                continue

            candidate_u = edge_u[collinear]
            candidate_v = edge_v[collinear]
            selected_coord_u = coord_u[collinear]
            selected_coord_v = coord_v[collinear]
            # Stable midpoint: unlike ``(u + v) / 2``, this cannot overflow
            # when a valid mesh uses very large but finite coordinates.
            candidate_coords = selected_coord_u + 0.5 * (
                selected_coord_v - selected_coord_u
            )
            station_ids, matched = _match_sorted_coordinates(
                candidate_coords,
                coordinates,
                eps,
            )
            if not np.any(matched):
                continue

            candidate_u = candidate_u[matched]
            candidate_v = candidate_v[matched]
            station_ids = station_ids[matched]
            low_nodes = np.minimum(candidate_u, candidate_v)
            high_nodes = np.maximum(candidate_u, candidate_v)
            encoded = np.column_stack((station_ids, low_nodes, high_nodes))
            # Interior element edges normally appear twice.  Chunk-local
            # deduplication halves retained memory for structured meshes.
            edge_chunks[axis].append(np.unique(encoded, axis=0))

    result = {}
    for axis in ("x", "y"):
        if edge_chunks[axis]:
            result[axis] = np.concatenate(edge_chunks[axis], axis=0)
        else:
            result[axis] = np.empty((0, 3), dtype=np.intp)
    return result


def _build_axis_station_topologies(
    nodes,
    retained_edges,
    station_coordinates,
    axis,
    eps,
):
    topologies = [None] * len(station_coordinates)
    if not len(retained_edges):
        return topologies

    order = np.argsort(retained_edges[:, 0], kind="stable")
    retained_edges = retained_edges[order]
    station_ids, starts = np.unique(
        retained_edges[:, 0],
        return_index=True,
    )
    ends = np.r_[starts[1:], len(retained_edges)]
    for station_id, start, end in zip(station_ids, starts, ends):
        topologies[int(station_id)] = _build_station_topology(
            nodes,
            retained_edges[start:end, 1:3],
            axis,
            station_coordinates[int(station_id)],
            eps,
        )
    return topologies


def _build_station_topology(nodes, edges, axis, coordinate, eps):
    """Build ordered connected components using actual shared node ids."""
    edges = np.unique(edges, axis=0)
    node_ids = np.unique(edges.ravel())
    span_axis = _AXIS_DATA[axis][1]
    span_values = nodes[node_ids, span_axis]
    order = np.lexsort((node_ids, span_values))
    node_ids = node_ids[order]
    span_values = span_values[order]

    if len(span_values) > 1 and np.any(np.diff(span_values) <= eps):
        raise ValueError(
            f"{axis}-axis station {coordinate} has duplicate or reversed "
            "structural nodes"
        )

    ids_numeric = np.sort(node_ids)
    span_position_by_numeric = np.argsort(node_ids)
    u_positions = span_position_by_numeric[
        np.searchsorted(ids_numeric, edges[:, 0])
    ]
    v_positions = span_position_by_numeric[
        np.searchsorted(ids_numeric, edges[:, 1])
    ]
    lower_positions = np.minimum(u_positions, v_positions)
    upper_positions = np.maximum(u_positions, v_positions)

    connected_to_next = np.zeros(max(len(node_ids) - 1, 0), dtype=bool)
    consecutive = upper_positions == lower_positions + 1
    connected_to_next[lower_positions[consecutive]] = True
    component_ids = np.zeros(len(node_ids), dtype=np.intp)
    if len(node_ids) > 1:
        component_ids[1:] = np.cumsum(~connected_to_next)

    return _StationTopology(
        node_ids=node_ids,
        span_values=span_values,
        component_ids=component_ids,
    )


def _validate_feature_edge_chains(
    rules,
    station_coordinates,
    station_topologies,
    eps,
):
    for rule in rules:
        axis = rule["axis"]
        station_id = _single_coordinate_index(
            station_coordinates[axis],
            rule["rail_coord"],
            eps,
        )
        topology = station_topologies[axis][station_id]
        description = (
            f"snap rule {rule['source_index']} on {axis}-axis rail "
            f"{rule['rail_id']} for span "
            f"[{rule['span_min']}, {rule['span_max']}]"
        )
        if topology is None:
            raise ValueError(f"mesh has no collinear edge chain for {description}")

        start_index = _optional_coordinate_index(
            topology.span_values,
            rule["span_min"],
            eps,
        )
        end_index = _optional_coordinate_index(
            topology.span_values,
            rule["span_max"],
            eps,
        )
        if start_index is None or end_index is None:
            raise ValueError(
                f"mesh edge chain is missing a feature endpoint for {description}"
            )
        if topology.component_ids[start_index] != topology.component_ids[end_index]:
            raise ValueError(f"mesh edge chain is disconnected for {description}")


def _validate_mandatory_stations(
    nodes,
    elements,
    mandatory,
    station_coordinates,
    station_topologies,
    *,
    eps,
    chunk_size,
):
    for axis, (coord_axis, _) in _AXIS_DATA.items():
        values = np.asarray(mandatory[axis], dtype=np.float64)
        if not len(values):
            continue

        for start in range(0, len(elements), chunk_size):
            chunk = elements[start : start + chunk_size]
            element_coords = nodes[chunk, coord_axis]
            minimum = np.min(element_coords, axis=1)
            maximum = np.max(element_coords, axis=1)
            candidate_ids = np.searchsorted(
                values,
                minimum + eps,
                side="right",
            )
            in_range = candidate_ids < len(values)
            if not np.any(in_range):
                continue
            rows = np.flatnonzero(in_range)
            candidates = values[candidate_ids[in_range]]
            crossing = candidates < maximum[in_range] - eps
            if np.any(crossing):
                local_row = rows[np.flatnonzero(crossing)[0]]
                coordinate = candidates[np.flatnonzero(crossing)[0]]
                raise ValueError(
                    f"mandatory {axis}-axis station {coordinate} crosses "
                    f"the interior of active element {start + int(local_row)}"
                )

        for coordinate in values:
            station_id = _single_coordinate_index(
                station_coordinates[axis],
                coordinate,
                eps,
            )
            topology = station_topologies[axis][station_id]
            if topology is None or len(topology.node_ids) < 2:
                raise ValueError(
                    f"mandatory {axis}-axis station {coordinate} has no "
                    "positive-length active mesh edge"
                )


def _match_sorted_coordinates(values, coordinates, eps):
    """Return nearest coordinate indices and a match mask for vector values."""
    right = np.searchsorted(coordinates, values, side="left")
    right_clipped = np.minimum(right, len(coordinates) - 1)
    left_clipped = np.maximum(right - 1, 0)
    right_distance = np.abs(values - coordinates[right_clipped])
    left_distance = np.abs(values - coordinates[left_clipped])
    choose_left = left_distance <= right_distance
    indices = np.where(choose_left, left_clipped, right_clipped)
    matched = np.abs(values - coordinates[indices]) <= eps
    return indices.astype(np.intp, copy=False), matched


def _single_coordinate_index(values, target, eps):
    index = _optional_coordinate_index(values, target, eps)
    if index is None:
        raise ValueError(f"required coordinate {target} is not indexed")
    return index


def _optional_coordinate_index(values, target, eps):
    if not len(values):
        return None
    indices, matched = _match_sorted_coordinates(
        np.asarray([target], dtype=np.float64),
        values,
        eps,
    )
    return int(indices[0]) if bool(matched[0]) else None


def _validate_coordinate_separation(coordinates, eps):
    if eps <= 0:
        return
    for axis, values in coordinates.items():
        if len(values) > 1 and np.any(np.diff(values) <= 2.0 * eps):
            raise ValueError(
                f"{axis}-axis required coordinates are ambiguous at eps={eps}"
            )


def _normalize_axis_coordinates(values, name):
    if values is None:
        return {"x": [], "y": []}
    if not isinstance(values, dict):
        raise ValueError(f"{name} must be a dictionary with x/y lists")
    if set(values) - {"x", "y"}:
        raise ValueError(f"{name} contains unsupported axes")
    return {
        axis: sorted(
            {
                _finite_number(value, f"{axis}-axis mandatory coordinate")
                for value in values.get(axis, ())
            }
        )
        for axis in ("x", "y")
    }


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _non_negative_finite_number(value, name):
    value = _finite_number(value, name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _finite_number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value
