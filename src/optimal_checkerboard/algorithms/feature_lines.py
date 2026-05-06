"""Extract pattern feature lines and shared rail metadata from faces."""

from optimal_checkerboard.algorithms.rail_builder import build_shared_rails


def _get_feature_lines(faces, element_size, return_details=False):
    """Extract shared checkerboard rails from supported face inputs.

    Args:
        faces: Face dictionaries using BOX, POLYGON, or LINE geometry.
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
            rail_data["restore_rules_by_z"],
            {
                "x": rail_data["x_rails"],
                "y": rail_data["y_rails"],
            },
        )

    return result


def _extract_lines(faces):
    """Convert supported face dictionaries into canonical pattern lines."""
    lines = []

    for face in faces:
        if face["type"] not in {"BOX", "LINE", "POLYGON"}:
            raise ValueError(f"Unsupported face type: {face['type']}")

        z_range = _face_z_range(face)
        if face["type"] == "BOX":
            lines.extend(_box_to_lines(face["dim"], z_range=z_range))
        elif face["type"] == "LINE":
            if not _is_flat_line_dim(face["dim"]):
                raise ValueError("LINE dim must be [x1,y1,x2,y2]")
            lines.append(_line_to_pattern_line(face["dim"], z_range=z_range))
        elif face["type"] == "POLYGON":
            for poly in _polygon_loops(face["dim"]):
                for i, point in enumerate(poly):
                    lines.append(
                        _line_to_pattern_line(
                            [poly[i - 1], point],
                            z_range=z_range,
                        )
                    )

    return lines


def _face_z_range(face):
    """Return an optional z range declared on a face dictionary."""
    if "bottom_z" not in face or "top_z" not in face:
        raise ValueError("Each face must define bottom_z and top_z")
    return _normalize_z_range([face["bottom_z"], face["top_z"]])


def _line_to_pattern_line(line, z_range=None):
    """Convert one supported line shape into [[x,y], [x,y], [z0,z1]]."""
    if _is_flat_line_dim(line):
        point1 = line[:2]
        point2 = line[2:]
    elif len(line) == 3 and len(line[0]) >= 2 and len(line[1]) >= 2:
        if z_range is None:
            if len(line[2]) != 2:
                raise ValueError("z_range must be [z_bottom, z_top]")
            z_range = line[2]
        point1, point2 = line[0], line[1]
    elif len(line) == 2:
        point1, point2 = line
    else:
        raise ValueError(
            "Lines must be [x1,y1,x2,y2], [[x1,y1,z], [x2,y2,z]], "
            "or [[x1,y1], [x2,y2], [z_bottom,z_top]]"
        )

    if z_range is None:
        z_range = _infer_flat_z_range(point1, point2, line)
    else:
        z_range = _normalize_z_range(z_range)

    return [
        [float(point1[0]), float(point1[1])],
        [float(point2[0]), float(point2[1])],
        z_range,
    ]


def _polygon_loops(dim):
    """Return polygon loops from flat or nested POLYGON dim input."""
    if not dim:
        raise ValueError("POLYGON dim must contain at least one point")
    if _is_point(dim[0]):
        return [dim]
    raise ValueError("POLYGON dim must be [[x1,y1], [x2,y2], ...]")


def _is_flat_line_dim(dim):
    """Return whether dim is [x1, y1, x2, y2]."""
    return len(dim) == 4 and all(not _is_sequence(value) for value in dim)


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


def _infer_flat_z_range(point1, point2, line, eps=0.01):
    """Infer a zero-height z range from legacy 3D endpoints."""
    if len(point1) < 3 or len(point2) < 3:
        raise ValueError("2D line endpoints require bottom_z/top_z")

    z1 = float(point1[2])
    z2 = float(point2[2])
    if abs(z1 - z2) > eps:
        raise ValueError(
            f"Z coordinates mismatch in line {line}: z1 and z2 must "
            "be equal."
        )
    return [z1, z1]


def _normalize_z_range(z_range):
    """Validate and float-normalize a z interval."""
    if len(z_range) != 2:
        raise ValueError("z range must be [bottom_z, top_z]")

    z_bottom = float(z_range[0])
    z_top = float(z_range[1])
    if z_top < z_bottom:
        raise ValueError("z_range must satisfy z_bottom <= z_top")
    return [z_bottom, z_top]


def _box_to_lines(dim, z_range=None):
    """Convert one rectangular box face definition into four pattern lines."""
    if len(dim) != 4:
        raise ValueError("BOX dim must be [x1,y1,x2,y2]")

    x1, y1, x2, y2 = dim
    z_range = _normalize_z_range(z_range)
    return [
        [[float(x1), float(y1)], [float(x2), float(y1)], list(z_range)],
        [[float(x2), float(y1)], [float(x2), float(y2)], list(z_range)],
        [[float(x2), float(y2)], [float(x1), float(y2)], list(z_range)],
        [[float(x1), float(y2)], [float(x1), float(y1)], list(z_range)],
    ]
