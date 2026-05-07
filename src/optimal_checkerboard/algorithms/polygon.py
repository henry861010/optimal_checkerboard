"""Helpers for POLYGON face loops."""


def normalize_polygon_loops(dim, eps=1e-12):
    """Validate POLYGON dim and classify loops by Cartesian winding.

    POLYGON dim is ``[[[x, y], ...], ...]``.  Clockwise loops are hulls and
    counter-clockwise loops are holes.
    """
    if not _is_sequence(dim) or len(dim) == 0:
        raise ValueError("POLYGON dim must contain at least one loop")
    if _is_point(dim[0]):
        raise ValueError("POLYGON dim must be [[[x1,y1], ...], ...]")

    loops = []
    has_hull = False
    for loop in dim:
        points = _normalize_loop_points(loop)
        points = _drop_repeated_closure(points, eps=eps)
        if len(points) < 3:
            raise ValueError("Each POLYGON loop must contain at least 3 points")

        area2 = polygon_signed_area2(points)
        if abs(area2) <= eps:
            raise ValueError("POLYGON loop area must be non-zero")

        role = "hull" if area2 < 0.0 else "hole"
        if role == "hull":
            has_hull = True
        loops.append(
            {
                "points": points,
                "area2": area2,
                "role": role,
            }
        )

    if not has_hull:
        raise ValueError("POLYGON dim must include at least one clockwise hull")
    return loops


def polygon_signed_area2(points):
    """Return twice the signed Cartesian polygon area."""
    area2 = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        area2 += point[0] * next_point[1] - next_point[0] * point[1]
    return area2


def _normalize_loop_points(loop):
    if not _is_sequence(loop) or len(loop) == 0:
        raise ValueError("Each POLYGON loop must be [[x1,y1], ...]")

    points = []
    for point in loop:
        if not _is_point(point):
            raise ValueError("Each POLYGON loop must be [[x1,y1], ...]")
        points.append([float(point[0]), float(point[1])])
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
