"""Spatial relation checks for axis-aligned line grouping."""


def _is_related(line1, line2, l, eps=0.01, v_h=0):
    """Return whether two same-axis lines are close and longitudinally apart.

    Args:
        line1: Reference 3D line in [[x1, y1, z1], [x2, y2, z2]] format.
        line2: Target 3D line in the same format.
        l: Distance used for lateral proximity and endpoint buffer checks.
        eps: Coordinate tolerance for floating-point comparisons.
        v_h: Axis selector; 0 for fixed x lines and 1 for fixed y lines.

    Returns:
        True when the lines satisfy the related-line test, otherwise False.
    """

    def get_info(line):
        """Extract fixed-axis position and varying-axis bounds from a line."""
        (x1, y1, z1), (x2, y2, z2) = line

        if abs(z1 - z2) > eps:
            raise ValueError("Lines must be on the same Z plane.")

        fixed_idx = v_h
        var_idx = 1 - v_h
        pos = line[0][fixed_idx]

        if abs(line[0][fixed_idx] - line[1][fixed_idx]) > eps:
            raise ValueError(
                "Line is not strictly aligned with the evaluated axis."
            )

        var_min = min(line[0][var_idx], line[1][var_idx])
        var_max = max(line[0][var_idx], line[1][var_idx])
        return pos, var_min, var_max

    try:
        pos1, min1, max1 = get_info(line1)
        pos2, min2, max2 = get_info(line2)
    except ValueError:
        return False

    if abs(pos1 - pos2) > l + eps:
        return False

    in_shaded_area_a = min2 >= max1 + l - eps
    in_shaded_area_b = max2 <= min1 - l + eps
    return in_shaded_area_a or in_shaded_area_b
