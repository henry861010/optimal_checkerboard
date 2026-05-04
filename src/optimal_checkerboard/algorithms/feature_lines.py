"""Extract pattern feature lines and shared rail metadata from faces."""

from optimal_checkerboard.algorithms.rail_builder import build_shared_rails


def _get_feature_lines(faces, element_size, return_details=False):
    """Extract shared checkerboard rails from box and polygon face inputs.

    Args:
        faces: Face dictionaries using either BOX or POLYGON geometry.
        element_size: Maximum coordinate gap used when merging nearby lines.
        return_details: Whether to include snap rules and rail metadata.

    Returns:
        Grouped vertical lines, grouped horizontal lines, x rails, y rails,
        and optionally snap rules plus serialized rail metadata.
    """
    lines = _extract_lines(faces)
    rail_data = build_shared_rails(lines, element_size)

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
            {
                "x": rail_data["x_rails"],
                "y": rail_data["y_rails"],
            },
        )

    return result


def _extract_lines(faces):
    """Convert supported face dictionaries into raw 3D line segments."""
    lines = []

    for face in faces:
        if face["type"] == "BOX":
            lines.extend(_box_to_lines(face["dim"]))
        elif face["type"] == "POLYGON":
            for poly in face["dim"]:
                for i, point in enumerate(poly):
                    lines.append([poly[i - 1], point])
        else:
            raise ValueError(f"Unsupported face type: {face['type']}")

    return lines


def _box_to_lines(dim):
    """Convert one rectangular box face definition into four edge lines."""
    x1, y1, z1, x2, y2, _ = dim
    return [
        [[x1, y1, z1], [x2, y1, z1]],
        [[x2, y1, z1], [x2, y2, z1]],
        [[x2, y2, z1], [x1, y2, z1]],
        [[x1, y2, z1], [x1, y1, z1]],
    ]
