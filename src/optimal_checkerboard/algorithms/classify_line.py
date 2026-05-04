"""Classify axis-aligned 3D pattern lines."""


def _classify_line(lines, eps=0.01):
    """Classify 3D lines into vertical and horizontal line lists.

    Args:
        lines: Lines formatted as [[x1, y1, z1], [x2, y2, z2]].
        eps: Coordinate tolerance for floating-point comparisons.

    Returns:
        A tuple containing vertical lines and horizontal lines.

    Raises:
        ValueError: If a line is not flat, is diagonal, or is a single point.
    """
    vertical_lines = []
    horizontal_lines = []

    for line in lines:
        (x1, y1, z1), (x2, y2, z2) = line

        if abs(z1 - z2) > eps:
            raise ValueError(
                f"Z coordinates mismatch in line {line}: z1 and z2 must "
                "be equal."
            )

        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        is_vertical = dx <= eps and dy > eps
        is_horizontal = dy <= eps and dx > eps

        if is_vertical:
            vertical_lines.append(line)
        elif is_horizontal:
            horizontal_lines.append(line)
        else:
            raise ValueError(
                f"Invalid line {line}: Must be strictly vertical or "
                "horizontal."
            )

    vertical_lines.sort(key=lambda line: (line[0][2], line[0][0]))
    horizontal_lines.sort(key=lambda line: (line[0][2], line[0][1]))

    return vertical_lines, horizontal_lines
