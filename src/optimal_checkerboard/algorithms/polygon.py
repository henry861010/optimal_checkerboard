"""Validation helpers for orthogonal POLYGON face loops."""

import math


def normalize_polygon_loops(dim, eps=0.0):
    """Validate POLYGON loops and classify them by Cartesian winding.

    POLYGON ``dim`` is ``[[[x, y], ...], ...]``.  Clockwise loops are
    disjoint hulls and counter-clockwise loops are holes.  The routine fails
    closed for diagonal edges, self intersections, intersecting loops, nested
    hulls, and holes that are not strictly contained by exactly one hull.
    """
    eps = float(eps)
    if not math.isfinite(eps) or eps < 0.0:
        raise ValueError("POLYGON eps must be finite and non-negative")
    if not _is_sequence(dim) or len(dim) == 0:
        raise ValueError("POLYGON dim must contain at least one loop")
    if _is_point(dim[0]):
        raise ValueError("POLYGON dim must be [[[x1,y1], ...], ...]")

    loops = []
    for loop_index, loop in enumerate(dim):
        points = _normalize_loop_points(loop)
        points = _drop_repeated_closure(points, eps=eps)
        if len(points) < 3:
            raise ValueError("Each POLYGON loop must contain at least 3 points")

        _validate_orthogonal_simple_loop(points, loop_index, eps)
        area2 = polygon_signed_area2(points)
        if not math.isfinite(area2) or abs(area2) <= eps:
            raise ValueError("POLYGON loop area must be finite and non-zero")

        loops.append(
            {
                "points": points,
                "area2": area2,
                "role": "hull" if area2 < 0.0 else "hole",
                "loop_index": loop_index,
            }
        )

    hull_indices = [
        index for index, loop in enumerate(loops) if loop["role"] == "hull"
    ]
    if not hull_indices:
        raise ValueError("POLYGON dim must include at least one clockwise hull")

    _validate_loop_relationships(loops, hull_indices, eps)
    return loops


def polygon_signed_area2(points):
    """Return twice the signed Cartesian polygon area."""
    origin_x, origin_y = points[0]
    area2 = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        point_x = point[0] - origin_x
        point_y = point[1] - origin_y
        next_x = next_point[0] - origin_x
        next_y = next_point[1] - origin_y
        area2 += point_x * next_y - next_x * point_y
    return area2


def _validate_orthogonal_simple_loop(points, loop_index, eps):
    segment_count = len(points)
    segments = []
    for point_index, point in enumerate(points):
        next_point = points[(point_index + 1) % segment_count]
        dx = next_point[0] - point[0]
        dy = next_point[1] - point[1]
        horizontal = abs(dy) <= eps
        vertical = abs(dx) <= eps
        if horizontal and vertical:
            raise ValueError(
                f"POLYGON loop {loop_index} contains a zero-length edge"
            )
        if not horizontal and not vertical:
            raise ValueError(
                f"POLYGON loop {loop_index} contains a non-orthogonal edge"
            )
        segments.append((point, next_point, "h" if horizontal else "v"))

    # Adjacent collinear segments may be redundant, but they must not reverse
    # and overlap each other.
    for index in range(segment_count):
        first = segments[index]
        second = segments[(index + 1) % segment_count]
        if first[2] != second[2]:
            continue
        incoming = (
            first[1][0] - first[0][0],
            first[1][1] - first[0][1],
        )
        outgoing = (
            second[1][0] - second[0][0],
            second[1][1] - second[0][1],
        )
        if incoming[0] * outgoing[0] + incoming[1] * outgoing[1] < 0.0:
            raise ValueError(
                f"POLYGON loop {loop_index} contains an overlapping backtrack"
            )

    for first_index, first in enumerate(segments):
        for second_index in range(first_index + 1, segment_count):
            if second_index == first_index + 1:
                continue
            if first_index == 0 and second_index == segment_count - 1:
                continue
            if _segments_intersect(first, segments[second_index], eps):
                raise ValueError(
                    f"POLYGON loop {loop_index} is self-intersecting"
                )


def _validate_loop_relationships(loops, hull_indices, eps):
    bboxes = [_loop_bbox(loop["points"]) for loop in loops]

    # Distinct loops may not cross or touch.  Bounding boxes cheaply discard
    # the common case before the segment-level test.
    for first_index, first in enumerate(loops):
        for second_index in range(first_index + 1, len(loops)):
            if not _bbox_intersects(
                bboxes[first_index],
                bboxes[second_index],
                eps,
            ):
                continue
            if _loops_intersect(
                first["points"],
                loops[second_index]["points"],
                eps,
            ):
                raise ValueError(
                    "POLYGON loops may not intersect or touch "
                    f"(loops {first_index} and {second_index})"
                )

    # Multiple hulls represent disjoint regions.  Nesting would make winding
    # semantics ambiguous and is rejected.
    for position, first_index in enumerate(hull_indices):
        for second_index in hull_indices[position + 1 :]:
            if (
                _point_location(
                    loops[first_index]["points"][0],
                    loops[second_index]["points"],
                    eps,
                )
                == 1
                or _point_location(
                    loops[second_index]["points"][0],
                    loops[first_index]["points"],
                    eps,
                )
                == 1
            ):
                raise ValueError("POLYGON hulls must be disjoint and non-nested")

    hole_indices = [
        index for index, loop in enumerate(loops) if loop["role"] == "hole"
    ]
    for hole_index in hole_indices:
        containing_hulls = [
            hull_index
            for hull_index in hull_indices
            if _point_location(
                loops[hole_index]["points"][0],
                loops[hull_index]["points"],
                eps,
            )
            == 1
        ]
        if len(containing_hulls) != 1:
            raise ValueError(
                f"POLYGON hole {hole_index} must be strictly contained by "
                "exactly one hull"
            )
        loops[hole_index]["hull_index"] = containing_hulls[0]

    # Nested holes do not describe a supported material region.  If islands
    # are needed they must be expressed as a separate, disjoint hull.
    for position, first_index in enumerate(hole_indices):
        for second_index in hole_indices[position + 1 :]:
            if loops[first_index]["hull_index"] != loops[second_index]["hull_index"]:
                continue
            if (
                _point_location(
                    loops[first_index]["points"][0],
                    loops[second_index]["points"],
                    eps,
                )
                == 1
                or _point_location(
                    loops[second_index]["points"][0],
                    loops[first_index]["points"],
                    eps,
                )
                == 1
            ):
                raise ValueError("POLYGON holes may not overlap or be nested")

    for hull_index in hull_indices:
        loops[hull_index]["hull_index"] = hull_index


def _loops_intersect(first_points, second_points, eps):
    first_segments = _loop_segments(first_points, eps)
    second_segments = _loop_segments(second_points, eps)
    for first in first_segments:
        first_bbox = _segment_bbox(first)
        for second in second_segments:
            if not _bbox_intersects(first_bbox, _segment_bbox(second), eps):
                continue
            if _segments_intersect(first, second, eps):
                return True
    return False


def _loop_segments(points, eps):
    segments = []
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        orientation = "h" if abs(next_point[1] - point[1]) <= eps else "v"
        segments.append((point, next_point, orientation))
    return segments


def _segments_intersect(first, second, eps):
    first_start, first_end, first_axis = first
    second_start, second_end, second_axis = second
    if first_axis == second_axis == "h":
        return (
            abs(first_start[1] - second_start[1]) <= eps
            and _closed_intervals_overlap(
                first_start[0],
                first_end[0],
                second_start[0],
                second_end[0],
                eps,
            )
        )
    if first_axis == second_axis == "v":
        return (
            abs(first_start[0] - second_start[0]) <= eps
            and _closed_intervals_overlap(
                first_start[1],
                first_end[1],
                second_start[1],
                second_end[1],
                eps,
            )
        )

    horizontal, vertical = (
        (first, second) if first_axis == "h" else (second, first)
    )
    horizontal_start, horizontal_end, _ = horizontal
    vertical_start, vertical_end, _ = vertical
    return (
        min(horizontal_start[0], horizontal_end[0]) - eps
        <= vertical_start[0]
        <= max(horizontal_start[0], horizontal_end[0]) + eps
        and min(vertical_start[1], vertical_end[1]) - eps
        <= horizontal_start[1]
        <= max(vertical_start[1], vertical_end[1]) + eps
    )


def _closed_intervals_overlap(first_a, first_b, second_a, second_b, eps):
    return max(min(first_a, first_b), min(second_a, second_b)) <= min(
        max(first_a, first_b),
        max(second_a, second_b),
    ) + eps


def _point_location(point, loop, eps):
    """Return 1 inside, 0 on the boundary, and -1 outside a loop."""
    for segment in _loop_segments(loop, eps):
        if _point_on_segment(point, segment, eps):
            return 0

    inside = False
    px, py = point
    for index, first in enumerate(loop):
        second = loop[(index + 1) % len(loop)]
        if (first[1] > py) == (second[1] > py):
            continue
        crossing_x = first[0] + (
            (py - first[1]) * (second[0] - first[0])
            / (second[1] - first[1])
        )
        if px < crossing_x:
            inside = not inside
    return 1 if inside else -1


def _point_on_segment(point, segment, eps):
    start, end, axis = segment
    if axis == "h":
        return (
            abs(point[1] - start[1]) <= eps
            and min(start[0], end[0]) - eps
            <= point[0]
            <= max(start[0], end[0]) + eps
        )
    return (
        abs(point[0] - start[0]) <= eps
        and min(start[1], end[1]) - eps
        <= point[1]
        <= max(start[1], end[1]) + eps
    )


def _loop_bbox(points):
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


def _segment_bbox(segment):
    start, end, _ = segment
    return (
        min(start[0], end[0]),
        min(start[1], end[1]),
        max(start[0], end[0]),
        max(start[1], end[1]),
    )


def _bbox_intersects(first, second, eps):
    return not (
        first[2] < second[0] - eps
        or second[2] < first[0] - eps
        or first[3] < second[1] - eps
        or second[3] < first[1] - eps
    )


def _normalize_loop_points(loop):
    if not _is_sequence(loop) or len(loop) == 0:
        raise ValueError("Each POLYGON loop must be [[x1,y1], ...]")

    points = []
    for point in loop:
        if not _is_point(point):
            raise ValueError("Each POLYGON loop must be [[x1,y1], ...]")
        x = float(point[0])
        y = float(point[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("POLYGON coordinates must be finite")
        points.append([x, y])
    return points


def _drop_repeated_closure(points, eps):
    if len(points) < 2:
        return points

    first = points[0]
    last = points[-1]
    if abs(first[0] - last[0]) <= eps and abs(first[1] - last[1]) <= eps:
        return points[:-1]
    return points


def _is_point(value):
    """Return whether value looks like one xy or xyz point."""
    return (
        _is_sequence(value)
        and len(value) >= 2
        and not _is_sequence(value[0])
        and not _is_sequence(value[1])
    )


def _is_sequence(value):
    """Return whether value behaves like a non-string sequence."""
    if isinstance(value, (str, bytes)):
        return False
    try:
        len(value)
    except TypeError:
        return False
    return True
