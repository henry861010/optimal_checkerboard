"""Classify axis-aligned pattern lines."""

import math


def _scale_aware_tolerance(values, requested=None):
    """Validate inputs and return an exact topology tolerance.

    Input floats are authoritative geometry.  Even a one-ULP difference is a
    representable non-axis-aligned edge and must not be silently projected
    onto an axis.  ``requested`` remains accepted for API compatibility, but
    it cannot loosen topology equality.
    """
    values = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("line coordinates and z values must be finite")
    if requested is None:
        return 0.0

    requested = float(requested)
    if not math.isfinite(requested) or requested < 0.0:
        raise ValueError("eps must be a non-negative finite number")
    return 0.0


def _line_components(line, eps=None):
    """Return x/y endpoints and z range for supported pattern line formats."""
    if (
        len(line) == 3
        and len(line[0]) >= 2
        and len(line[1]) >= 2
        and len(line[2]) == 2
    ):
        z_bottom = float(line[2][0])
        z_top = float(line[2][1])
        components = (
            float(line[0][0]),
            float(line[0][1]),
            float(line[1][0]),
            float(line[1][1]),
            z_bottom,
            z_top,
        )
        if not all(math.isfinite(value) for value in components):
            raise ValueError("line coordinates and z values must be finite")
        if z_top < z_bottom:
            raise ValueError("z_range must satisfy z_bottom <= z_top")
        return components

    if len(line) != 2:
        raise ValueError(
            "Lines must be [[x1,y1,z], [x2,y2,z]] or "
            "[[x1,y1], [x2,y2], [z_bottom,z_top]]"
        )

    point1, point2 = line
    if len(point1) < 3 or len(point2) < 3:
        raise ValueError("2D line endpoints require z_range")

    z1 = float(point1[2])
    z2 = float(point2[2])
    components = (
        float(point1[0]),
        float(point1[1]),
        float(point2[0]),
        float(point2[1]),
        z1,
        z2,
    )
    if not all(math.isfinite(value) for value in components):
        raise ValueError("line coordinates and z values must be finite")
    z_tol = _scale_aware_tolerance((z1, z2), requested=eps)
    if abs(z1 - z2) > z_tol:
        raise ValueError(
            f"Z coordinates mismatch in line {line}: z1 and z2 must "
            "be equal."
        )

    return components[:4] + (z1, z1)


def _classify_line(lines, eps=None):
    """Classify pattern lines into vertical and horizontal line lists.

    Args:
        lines: Lines formatted as [[x1,y1,z], [x2,y2,z]] or
            [[x1,y1], [x2,y2], [z_bottom,z_top]].
        eps: Retained for API compatibility.  It is validated but cannot
            reinterpret two distinct floating-point coordinates as equal.

    Returns:
        A tuple containing vertical lines and horizontal lines.

    Raises:
        ValueError: If a line is not flat, is diagonal, or is a single point.
    """
    vertical_lines = []
    horizontal_lines = []

    for line in lines:
        x1, y1, x2, y2, _, _ = _line_components(line, eps=eps)

        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        xy_tol = _scale_aware_tolerance(
            (x1, y1, x2, y2),
            requested=eps,
        )
        is_vertical = dx <= xy_tol and dy > xy_tol
        is_horizontal = dy <= xy_tol and dx > xy_tol

        if is_vertical:
            vertical_lines.append(line)
        elif is_horizontal:
            horizontal_lines.append(line)
        else:
            raise ValueError(
                f"Invalid line {line}: Must be strictly vertical or "
                "horizontal."
            )

    vertical_lines.sort(
        key=lambda line: (
            _line_components(line, eps=eps)[4],
            _line_components(line, eps=eps)[0],
        )
    )
    horizontal_lines.sort(
        key=lambda line: (
            _line_components(line, eps=eps)[4],
            _line_components(line, eps=eps)[1],
        )
    )

    return vertical_lines, horizontal_lines
