"""Group line segments by rounded z coordinate."""

from collections import defaultdict


def _group_lines_by_z(lines, decimal_places=4):
    """Group 3D line segments by rounded first-node z coordinate."""
    grouped_lines = defaultdict(list)

    for line in lines:
        node1, _ = line
        z_rounded = round(node1[2], decimal_places)
        grouped_lines[z_rounded].append(line)

    return dict(sorted(grouped_lines.items()))
