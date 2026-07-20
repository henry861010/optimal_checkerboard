"""Extract pattern feature lines and shared rail metadata from faces."""

import math

from optimal_checkerboard.algorithms.classify_line import (
    _classify_line,
    _line_components,
)
from optimal_checkerboard.algorithms.polygon import normalize_polygon_loops
from optimal_checkerboard.algorithms.rail_builder import build_shared_rails


def _get_feature_lines(
    faces,
    element_size,
    return_details=False,
    pinned_coords=None,
):
    """Extract shared checkerboard rails from supported face inputs.

    Args:
        faces: Face dictionaries using BOX, POLYGON, or LINE geometry.
        element_size: Maximum coordinate gap used when merging nearby lines.
        return_details: Whether to include snap rules and rail metadata.

    Returns:
        Grouped vertical lines, grouped horizontal lines, x rails, y rails,
        and optionally snap rules plus serialized rail metadata.
    """
    lines = _extract_lines(faces)
    rail_data = build_shared_rails(
        lines,
        element_size,
        pinned_coords=pinned_coords,
    )

    group_lines_v = [rail["lines_by_z"] for rail in rail_data["x_rails"]]
    group_lines_h = [rail["lines_by_z"] for rail in rail_data["y_rails"]]
    result = (
        group_lines_v,
        group_lines_h,
        rail_data["x_list"],
        rail_data["y_list"],
    )

    if return_details:
        return result + (
            rail_data["snap_rules_by_z"],
            rail_data["restore_rules_by_z"],
            {
                "x": rail_data["x_rails"],
                "y": rail_data["y_rails"],
            },
        )

    return result


def _extract_lines(faces):
    """Convert supported face dictionaries into canonical pattern lines."""
    lines = []

    for face in faces:
        if face["type"] not in {"BOX", "LINE", "POLYGON"}:
            raise ValueError(f"Unsupported face type: {face['type']}")

        z_range = _face_z_range(face)
        if face["type"] == "BOX":
            lines.extend(_box_to_lines(face["dim"], z_range=z_range))
        elif face["type"] == "LINE":
            if not _is_flat_line_dim(face["dim"]):
                raise ValueError("LINE dim must be [x1,y1,x2,y2]")
            lines.append(_line_to_pattern_line(face["dim"], z_range=z_range))
        elif face["type"] == "POLYGON":
            for poly in _polygon_loops(face["dim"]):
                for i, point in enumerate(poly):
                    lines.append(
                        _line_to_pattern_line(
                            [poly[i - 1], point],
                            z_range=z_range,
                        )
                    )

    return lines


def feature_span_endpoint_coordinates(faces, eps=0.0):
    """Return cross-axis stations required by every finite feature line.

    A vertical feature contributes its fixed x coordinate to the shared x
    rails, while its y span endpoints must independently be present on the y
    mesh axis.  The converse applies to horizontal features.  Collecting these
    stations from the canonical feature lines covers BOX, POLYGON, and LINE
    faces without work proportional to the generated mesh size.

    The result is ``{"x": [...], "y": [...]}``, where x values come from
    horizontal span endpoints and y values come from vertical span endpoints.
    """
    try:
        eps = float(eps)
    except (TypeError, ValueError) as exc:
        raise ValueError("eps must be a non-negative finite number") from exc
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("eps must be a non-negative finite number")

    lines = _extract_lines(faces)
    vertical_lines, horizontal_lines = _classify_line(lines, eps=eps)
    mandatory = {"x": set(), "y": set()}
    for line in vertical_lines:
        _, y1, _, y2, _, _ = _line_components(line, eps=eps)
        mandatory["y"].update((y1, y2))
    for line in horizontal_lines:
        x1, _, x2, _, _, _ = _line_components(line, eps=eps)
        mandatory["x"].update((x1, x2))

    return {
        axis: sorted(values)
        for axis, values in mandatory.items()
    }


def line_span_endpoint_coordinates(faces, eps=0.0):
    """Backward-compatible alias for feature span endpoint collection."""
    return feature_span_endpoint_coordinates(faces, eps=eps)


def static_feature_span_endpoint_coordinates(faces):
    """Return endpoints that cannot be represented by coupled snap rails.

    A feature endpoint is dynamic only when exact perpendicular features
    cross that point and their z-interval union covers the feature's complete
    active interval.  Closed BOX/POLYGON corners therefore remain eligible
    for rail sharing, while isolated LINE endpoints and partially covered
    corners become structural coordinates.
    """
    lines = _extract_lines(faces)
    vertical_lines, horizontal_lines = _classify_line(lines)
    vertical_features = [
        _static_endpoint_feature(line, "x")
        for line in vertical_lines
    ]
    horizontal_features = [
        _static_endpoint_feature(line, "y")
        for line in horizontal_lines
    ]

    vertical_by_coord = _features_by_coord(vertical_features)
    horizontal_by_coord = _features_by_coord(horizontal_features)
    static_coords = {"x": set(), "y": set()}
    coverage_cache = {}

    for feature in vertical_features:
        for endpoint_coord in (feature["span_min"], feature["span_max"]):
            if not _perpendicular_features_cover_endpoint(
                feature,
                endpoint_coord,
                horizontal_by_coord,
                coverage_cache,
            ):
                static_coords["y"].add(endpoint_coord)

    for feature in horizontal_features:
        for endpoint_coord in (feature["span_min"], feature["span_max"]):
            if not _perpendicular_features_cover_endpoint(
                feature,
                endpoint_coord,
                vertical_by_coord,
                coverage_cache,
            ):
                static_coords["x"].add(endpoint_coord)

    return {
        axis: sorted(coords)
        for axis, coords in static_coords.items()
    }


def _static_endpoint_feature(line, axis):
    """Normalize one canonical line for endpoint coverage checks."""
    x1, y1, x2, y2, z_bottom, z_top = _line_components(line)
    if axis == "x":
        coord = x1
        span_min, span_max = sorted((y1, y2))
    else:
        coord = y1
        span_min, span_max = sorted((x1, x2))
    return {
        "axis": axis,
        "coord": float(coord),
        "span_min": float(span_min),
        "span_max": float(span_max),
        "z_bottom": float(z_bottom),
        "z_top": float(z_top),
    }


def _features_by_coord(features):
    """Index perpendicular-feature candidates by exact target coordinate."""
    result = {}
    for feature in features:
        result.setdefault(feature["coord"], []).append(feature)
    return result


def _perpendicular_features_cover_endpoint(
    feature,
    endpoint_coord,
    perpendicular_by_coord,
    coverage_cache,
):
    """Return whether perpendicular z intervals cover one full endpoint."""
    cache_key = (
        feature["axis"],
        feature["coord"],
        endpoint_coord,
        feature["z_bottom"],
        feature["z_top"],
    )
    cached = coverage_cache.get(cache_key)
    if cached is not None:
        return cached

    intervals = [
        (other["z_bottom"], other["z_top"])
        for other in perpendicular_by_coord.get(endpoint_coord, ())
        if other["span_min"] <= feature["coord"] <= other["span_max"]
    ]
    covered = _z_intervals_cover(
        intervals,
        feature["z_bottom"],
        feature["z_top"],
    )
    coverage_cache[cache_key] = covered
    return covered


def _z_intervals_cover(intervals, required_bottom, required_top):
    """Return whether closed intervals cover one complete closed interval."""
    cursor = required_bottom
    for bottom, top in sorted(intervals):
        if top < cursor:
            continue
        if bottom > cursor:
            return False
        cursor = max(cursor, top)
        if cursor >= required_top:
            return True
    return False


def _face_z_range(face):
    """Return an optional z range declared on a face dictionary."""
    if "bottom_z" not in face or "top_z" not in face:
        raise ValueError("Each face must define bottom_z and top_z")
    return _normalize_z_range([face["bottom_z"], face["top_z"]])


def _line_to_pattern_line(line, z_range=None):
    """Convert one supported line shape into [[x,y], [x,y], [z0,z1]]."""
    if _is_flat_line_dim(line):
        point1 = line[:2]
        point2 = line[2:]
    elif len(line) == 3 and len(line[0]) >= 2 and len(line[1]) >= 2:
        if z_range is None:
            if len(line[2]) != 2:
                raise ValueError("z_range must be [z_bottom, z_top]")
            z_range = line[2]
        point1, point2 = line[0], line[1]
    elif len(line) == 2:
        point1, point2 = line
    else:
        raise ValueError(
            "Lines must be [x1,y1,x2,y2], [[x1,y1,z], [x2,y2,z]], "
            "or [[x1,y1], [x2,y2], [z_bottom,z_top]]"
        )

    if z_range is None:
        z_range = _infer_flat_z_range(point1, point2, line)
    else:
        z_range = _normalize_z_range(z_range)

    coordinates = [
        float(point1[0]),
        float(point1[1]),
        float(point2[0]),
        float(point2[1]),
    ]
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError("feature-line coordinates must be finite")
    return [coordinates[:2], coordinates[2:], z_range]


def _polygon_loops(dim):
    """Return POLYGON loops from canonical nested dim input."""
    return [loop["points"] for loop in normalize_polygon_loops(dim)]


def _is_flat_line_dim(dim):
    """Return whether dim is [x1, y1, x2, y2]."""
    return len(dim) == 4 and all(not _is_sequence(value) for value in dim)


def _is_sequence(value):
    """Return whether value behaves like a non-string sequence."""
    if isinstance(value, (str, bytes)):
        return False
    try:
        len(value)
    except TypeError:
        return False
    return True


def _infer_flat_z_range(point1, point2, line, eps=0.0):
    """Infer a zero-height z range from legacy 3D endpoints."""
    if len(point1) < 3 or len(point2) < 3:
        raise ValueError("2D line endpoints require bottom_z/top_z")

    z1 = float(point1[2])
    z2 = float(point2[2])
    if not math.isfinite(z1) or not math.isfinite(z2):
        raise ValueError("z coordinates must be finite")
    if abs(z1 - z2) > eps:
        raise ValueError(
            f"Z coordinates mismatch in line {line}: z1 and z2 must "
            "be equal."
        )
    return [z1, z1]


def _normalize_z_range(z_range):
    """Validate and float-normalize a z interval."""
    if len(z_range) != 2:
        raise ValueError("z range must be [bottom_z, top_z]")

    z_bottom = float(z_range[0])
    z_top = float(z_range[1])
    if not math.isfinite(z_bottom) or not math.isfinite(z_top):
        raise ValueError("z range values must be finite")
    if z_top < z_bottom:
        raise ValueError("z_range must satisfy z_bottom <= z_top")
    return [z_bottom, z_top]


def _box_to_lines(dim, z_range=None):
    """Convert one rectangular box face definition into four pattern lines."""
    if len(dim) != 4:
        raise ValueError("BOX dim must be [x1,y1,x2,y2]")

    x1, y1, x2, y2 = [float(value) for value in dim]
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        raise ValueError("BOX coordinates must be finite")
    xmin, xmax = sorted((x1, x2))
    ymin, ymax = sorted((y1, y2))
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("BOX dim must have positive area")
    z_range = _normalize_z_range(z_range)
    return [
        [[xmin, ymin], [xmax, ymin], list(z_range)],
        [[xmax, ymin], [xmax, ymax], list(z_range)],
        [[xmax, ymax], [xmin, ymax], list(z_range)],
        [[xmin, ymax], [xmin, ymin], list(z_range)],
    ]
