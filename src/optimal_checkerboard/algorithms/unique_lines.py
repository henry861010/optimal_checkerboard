"""Remove duplicate line segments from grouped line dictionaries."""


def _unique_lines(group_lines, decimals=4):
    """Remove duplicate 3D lines while preserving the first occurrence.

    Args:
        group_lines: Mapping from z level to line lists.
        decimals: Number of decimal places used for coordinate keys.

    Returns:
        A simplified mapping that omits duplicate lines and empty z groups.
    """
    seen_lines = set()
    group_lines_simplified = {}

    for z, lines in group_lines.items():
        unique_for_z = []
        for line in lines:
            line_key = _line_key(line, decimals)
            if line_key not in seen_lines:
                seen_lines.add(line_key)
                unique_for_z.append(line)

        if unique_for_z:
            group_lines_simplified[z] = unique_for_z

    return group_lines_simplified


def _line_key(line, decimals):
    """Return a direction-independent rounded key for one 3D line."""
    (x1, y1, _), (x2, y2, _) = line
    p1 = (round(x1, decimals), round(y1, decimals))
    p2 = (round(x2, decimals), round(y2, decimals))
    return tuple(sorted([p1, p2]))
