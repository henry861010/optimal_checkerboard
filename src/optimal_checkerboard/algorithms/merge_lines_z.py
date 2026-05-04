"""Merge line groups across z layers."""

import numpy as np


def _merge_lines_z(groups_z, element_size, v_h=0):
    """Group spatially close line sets across different z levels.

    Args:
        groups_z: Mapping from z value to grouped line sets.
        element_size: Maximum allowed distance between group centers.
        v_h: Axis selector; 0 for x-axis grouping and 1 for y-axis grouping.

    Returns:
        A tuple containing grouped line dictionaries and their coordinate
        bounds along the selected axis.
    """
    group_lines = []
    group_bounds = []

    for z, line_sets in groups_z.items():
        for lines in line_sets:
            lines_arr = np.array(lines, dtype=np.float64)
            min_v = np.min(lines_arr[:, :, v_h])
            max_v = np.max(lines_arr[:, :, v_h])
            center_v = (max_v + min_v) / 2.0

            is_merged = False
            for index, group in enumerate(group_lines):
                if z in group:
                    continue

                g_min_v, g_max_v = group_bounds[index]
                g_center_v = (g_max_v + g_min_v) / 2.0

                if abs(center_v - g_center_v) < element_size:
                    group[z] = lines_arr
                    group_bounds[index][0] = min(g_min_v, min_v)
                    group_bounds[index][1] = max(g_max_v, max_v)
                    is_merged = True
                    break

            if not is_merged:
                group_lines.append({z: lines_arr})
                group_bounds.append([min_v, max_v])

    return group_lines, group_bounds
