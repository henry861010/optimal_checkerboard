"""Convert pattern faces onto shared-rail default coordinates."""

from copy import deepcopy

from optimal_checkerboard.algorithms.classify_line import (
    _scale_aware_tolerance,
)


def snap_faces_to_shared_rails(faces, snap_rules_by_z, eps=0.0):
    """Return face copies whose snapped coordinates use shared rail defaults."""
    return [
        _snap_face_to_rail_defaults(
            face,
            snap_rules_by_z,
            eps=eps,
        )
        for face in faces
    ]


def _snap_face_to_rail_defaults(face, snap_rules_by_z, eps=0.0):
    """Return one face copy with snap-rule coordinates on shared rails."""
    face_dup = deepcopy(face)
    face_type = face["type"]

    if face_type == "BOX":
        face_dup["dim"] = _snap_box_dim_to_rail_defaults(
            face["dim"],
            face["bottom_z"],
            face["top_z"],
            snap_rules_by_z,
            eps=eps,
        )
    elif face_type == "LINE":
        face_dup["dim"] = _snap_line_dim_to_rail_defaults(
            face["dim"],
            face["bottom_z"],
            face["top_z"],
            snap_rules_by_z,
            eps=eps,
        )
    elif face_type == "POLYGON":
        face_dup["dim"] = _snap_polygon_dim_to_rail_defaults(
            face["dim"],
            face["bottom_z"],
            face["top_z"],
            snap_rules_by_z,
            eps=eps,
        )
    else:
        raise ValueError(f"Unsupported face type: {face_type}")

    return face_dup


def _snap_box_dim_to_rail_defaults(
    dim,
    bottom_z,
    top_z,
    snap_rules_by_z,
    eps=0.0,
):
    """Move BOX boundary coordinates to matching shared rails."""
    if len(dim) != 4:
        raise ValueError("BOX dim must be [x1,y1,x2,y2]")

    x1, y1, x2, y2 = [float(value) for value in dim]
    snapped_x1 = _snap_edge_coord_to_rail_default(
        "x",
        x1,
        y1,
        y2,
        bottom_z,
        top_z,
        snap_rules_by_z,
        eps=eps,
    )
    snapped_x2 = _snap_edge_coord_to_rail_default(
        "x",
        x2,
        y1,
        y2,
        bottom_z,
        top_z,
        snap_rules_by_z,
        eps=eps,
    )
    snapped_y1 = _snap_edge_coord_to_rail_default(
        "y",
        y1,
        x1,
        x2,
        bottom_z,
        top_z,
        snap_rules_by_z,
        eps=eps,
    )
    snapped_y2 = _snap_edge_coord_to_rail_default(
        "y",
        y2,
        x1,
        x2,
        bottom_z,
        top_z,
        snap_rules_by_z,
        eps=eps,
    )
    return [snapped_x1, snapped_y1, snapped_x2, snapped_y2]


def _snap_line_dim_to_rail_defaults(
    dim,
    bottom_z,
    top_z,
    snap_rules_by_z,
    eps=0.0,
):
    """Move a LINE coordinate to its matching shared rail."""
    if len(dim) != 4:
        raise ValueError("LINE dim must be [x1,y1,x2,y2]")

    x1, y1, x2, y2 = [float(value) for value in dim]
    edge = _snap_edge_axis_coord_to_rail_default(
        [x1, y1],
        [x2, y2],
        bottom_z,
        top_z,
        snap_rules_by_z,
        eps=eps,
    )
    if edge is None:
        return [x1, y1, x2, y2]

    axis, snapped_coord = edge
    if axis == "x":
        x1 = snapped_coord
        x2 = snapped_coord
    else:
        y1 = snapped_coord
        y2 = snapped_coord
    return [x1, y1, x2, y2]


def _snap_polygon_dim_to_rail_defaults(
    dim,
    bottom_z,
    top_z,
    snap_rules_by_z,
    eps=0.0,
):
    """Move POLYGON loop vertices to matching shared rails."""
    snapped_loops = []

    for loop in dim:
        snapped_loop = [
            [float(point[0]), float(point[1])]
            for point in loop
        ]
        if not snapped_loop:
            snapped_loops.append(snapped_loop)
            continue

        point_count = len(loop)
        closure_tol = _scale_aware_tolerance(
            (
                float(loop[0][0]),
                float(loop[0][1]),
                float(loop[-1][0]),
                float(loop[-1][1]),
            ),
            requested=eps,
        )
        has_repeated_closure = (
            point_count > 1
            and abs(float(loop[0][0]) - float(loop[-1][0])) <= closure_tol
            and abs(float(loop[0][1]) - float(loop[-1][1])) <= closure_tol
        )
        edge_point_count = (
            point_count - 1
            if has_repeated_closure
            else point_count
        )

        for point_index in range(edge_point_count):
            prev_index = (
                edge_point_count - 1
                if point_index == 0
                else point_index - 1
            )
            edge = _snap_edge_axis_coord_to_rail_default(
                loop[prev_index],
                loop[point_index],
                bottom_z,
                top_z,
                snap_rules_by_z,
                eps=eps,
            )
            if edge is None:
                continue

            axis, snapped_coord = edge
            coord_index = 0 if axis == "x" else 1
            snapped_loop[prev_index][coord_index] = snapped_coord
            snapped_loop[point_index][coord_index] = snapped_coord

        if has_repeated_closure:
            snapped_loop[-1] = list(snapped_loop[0])

        snapped_loops.append(snapped_loop)

    return snapped_loops


def _snap_edge_axis_coord_to_rail_default(
    point1,
    point2,
    bottom_z,
    top_z,
    snap_rules_by_z,
    eps=0.0,
):
    """Return ``(axis, rail_coord)`` for an orthogonal edge."""
    x1, y1 = float(point1[0]), float(point1[1])
    x2, y2 = float(point2[0]), float(point2[1])
    topology_tol = _scale_aware_tolerance(
        (x1, y1, x2, y2),
        requested=eps,
    )

    if abs(x1 - x2) <= topology_tol:
        return (
            "x",
            _snap_edge_coord_to_rail_default(
                "x",
                x1,
                y1,
                y2,
                bottom_z,
                top_z,
                snap_rules_by_z,
                eps=eps,
            ),
        )
    if abs(y1 - y2) <= topology_tol:
        return (
            "y",
            _snap_edge_coord_to_rail_default(
                "y",
                y1,
                x1,
                x2,
                bottom_z,
                top_z,
                snap_rules_by_z,
                eps=eps,
            ),
        )

    return None


def _snap_edge_coord_to_rail_default(
    axis,
    coord,
    span_start,
    span_end,
    bottom_z,
    top_z,
    snap_rules_by_z,
    eps=0.0,
):
    """Return shared rail coord for an edge, or the original coord."""
    span_min = min(float(span_start), float(span_end))
    span_max = max(float(span_start), float(span_end))
    coord = float(coord)
    bottom_z = _snap_rule_z_key(bottom_z)
    top_z = _snap_rule_z_key(top_z)

    for rule in _iter_snap_rules(snap_rules_by_z):
        if rule["axis"] != axis:
            continue
        if float(rule["target_coord"]) != coord:
            continue
        if float(rule["span_min"]) != span_min:
            continue
        if float(rule["span_max"]) != span_max:
            continue
        if float(rule["z_bottom"]) != bottom_z:
            continue
        if float(rule["z_top"]) != top_z:
            continue
        return float(rule["rail_coord"])

    return coord


def _snap_rule_z_key(value):
    """Return the unrounded z value used by shared-rail snap rules."""
    return float(value)


def _iter_snap_rules(snap_rules_by_z):
    """Yield snap rules from a z-indexed mapping."""
    if not snap_rules_by_z:
        return

    for rules in snap_rules_by_z.values():
        yield from rules
