"""Merge nearby same-layer lines in the xy plane."""

from optimal_checkerboard.algorithms.is_related import _is_related


def _merge_lines_xy(lines, element_size, v_h=0, eps=0.01):
    """Group nearby same-axis lines on one z plane.

    Args:
        lines: 3D lines formatted as [[x1, y1, z1], [x2, y2, z2]].
        element_size: Distance threshold for grouping related lines.
        v_h: Axis selector; 0 for fixed x lines and 1 for fixed y lines.
        eps: Coordinate tolerance for floating-point comparisons.

    Returns:
        A list of line groups.
    """
    if not lines:
        return []

    lines = sorted(lines, key=lambda x: x[0][v_h])
    line_count = len(lines)

    if line_count == 1:
        return [[lines[0]]]
    if line_count == 2:
        return [[lines[0]], [lines[1]]]

    result_groups = [[lines[0]], [lines[1]]]

    for i in range(2, line_count - 1):
        current_line = lines[i]
        last_group = result_groups[-1]
        is_related_to_group = any(
            _is_related(current_line, member, element_size, eps, v_h)
            for member in last_group
        )

        if is_related_to_group:
            last_group.append(current_line)
        else:
            result_groups.append([current_line])

    result_groups.append([lines[-1]])
    return sorted(result_groups, key=lambda group: group[0][0][v_h])
