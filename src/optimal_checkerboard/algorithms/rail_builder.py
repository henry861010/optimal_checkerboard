"""Build shared checkerboard rails and per-layer snap rules."""

from collections import defaultdict

from optimal_checkerboard.algorithms.classify_line import (
    _classify_line,
    _line_components,
)


def _round_key(value, decimals=4):
    """Round a coordinate into a stable dictionary key."""
    return round(float(value), decimals)


def _line_to_feature(line, axis, feature_id, eps=0.01, z_decimals=4):
    """Convert one geometric line into a normalized feature dictionary."""
    x1, y1, x2, y2, z_bottom, z_top = _line_components(line, eps=eps)

    if axis == "x":
        coord = (x1 + x2) / 2.0
        span_min = min(y1, y2)
        span_max = max(y1, y2)
    elif axis == "y":
        coord = (y1 + y2) / 2.0
        span_min = min(x1, x2)
        span_max = max(x1, x2)
    else:
        raise ValueError("axis must be 'x' or 'y'")

    z_bottom = _round_key(z_bottom, z_decimals)
    z_top = _round_key(z_top, z_decimals)
    return {
        "feature_id": feature_id,
        "axis": axis,
        "coord": float(coord),
        "z": z_bottom,
        "z_bottom": z_bottom,
        "z_top": z_top,
        "span_min": float(span_min),
        "span_max": float(span_max),
        "line": [
            [float(x1), float(y1)],
            [float(x2), float(y2)],
            [float(z_bottom), float(z_top)],
        ],
    }


def _spans_overlap_or_touch(a_min, a_max, b_min, b_max, eps):
    """Return whether two one-dimensional spans overlap or touch."""
    return max(a_min, b_min) <= min(a_max, b_max) + eps


def _z_ranges_active_overlap(feature, other, eps):
    """Return whether two features are active over the same z interval.

    Positive-height ranges conflict only over positive overlap.  Zero-height
    legacy features still conflict with another feature at the same snap event.
    """
    a_min = feature["z_bottom"]
    a_max = feature["z_top"]
    b_min = other["z_bottom"]
    b_max = other["z_top"]
    a_zero = abs(a_max - a_min) <= eps
    b_zero = abs(b_max - b_min) <= eps

    if a_zero and b_zero:
        return abs(a_min - b_min) <= eps
    if a_zero:
        return b_min - eps <= a_min < b_max - eps
    if b_zero:
        return a_min - eps <= b_min < a_max - eps
    return max(a_min, b_min) < min(a_max, b_max) - eps


def _can_add_to_rail(rail, feature, merge_tol, eps, all_features):
    """Return whether a feature is compatible with an existing shared rail."""
    new_min = min(rail["min_coord"], feature["coord"])
    new_max = max(rail["max_coord"], feature["coord"])
    if new_max - new_min > merge_tol + eps:
        return False

    # Overlapping active z intervals cannot share one rail over the same span
    # if they snap to different target coordinates.
    for member in rail["members"]:
        if not _z_ranges_active_overlap(member, feature, eps):
            continue
        if (
            _spans_overlap_or_touch(
                member["span_min"],
                member["span_max"],
                feature["span_min"],
                feature["span_max"],
                eps,
            )
            and abs(member["coord"] - feature["coord"]) > eps
        ):
            return False

    if _has_near_endpoint_conflict(rail, feature, merge_tol, eps):
        return False

    if _has_blocking_intermediate_feature(
        rail,
        feature,
        new_min,
        new_max,
        all_features,
        eps,
    ):
        return False

    return True


def _has_near_endpoint_conflict(rail, feature, merge_tol, eps):
    """Return whether near active endpoints should prevent rail sharing."""
    for member in rail["members"]:
        if not _z_ranges_active_overlap(member, feature, eps):
            continue
        if abs(member["coord"] - feature["coord"]) <= eps:
            continue

        span_gap = _span_gap(
            member["span_min"],
            member["span_max"],
            feature["span_min"],
            feature["span_max"],
        )
        if eps < span_gap < merge_tol - eps:
            return True

    return False


def _span_gap(a_min, a_max, b_min, b_max):
    """Return the positive gap between disjoint spans, or zero otherwise."""
    if a_max < b_min:
        return b_min - a_max
    if b_max < a_min:
        return a_min - b_max
    return 0.0


def _has_blocking_intermediate_feature(
    rail,
    feature,
    new_min,
    new_max,
    all_features,
    eps,
):
    """Return whether an active intermediate feature blocks a rail merge.

    A feature whose coordinate lies between the proposed rail bounds acts as
    an ordering barrier when its active z range and span overlap the new
    feature or an existing rail member.
    """
    member_ids = {member["feature_id"] for member in rail["members"]}
    candidate_ids = member_ids | {feature["feature_id"]}
    candidates = [feature] + rail["members"]

    for other in all_features:
        if other["feature_id"] in candidate_ids:
            continue
        if not new_min + eps < other["coord"] < new_max - eps:
            continue
        if _feature_overlaps_any(other, candidates, eps):
            return True

    return False


def _feature_overlaps_any(feature, others, eps):
    """Return whether a feature overlaps another in z and span."""
    for other in others:
        if not _z_ranges_active_overlap(feature, other, eps):
            continue
        if _spans_overlap_or_touch(
            feature["span_min"],
            feature["span_max"],
            other["span_min"],
            other["span_max"],
            eps,
        ):
            return True
    return False


def _add_to_rail(rail, feature):
    """Append a feature to a rail and update rail bounds and z indexes."""
    rail["members"].append(feature)
    rail["min_coord"] = min(rail["min_coord"], feature["coord"])
    rail["max_coord"] = max(rail["max_coord"], feature["coord"])
    rail["coord"] = (rail["min_coord"] + rail["max_coord"]) / 2.0
    rail["lines_by_z"][feature["z"]].append(feature["line"])
    rail["spans_by_z"][feature["z"]].append(
        (feature["span_min"], feature["span_max"], feature["coord"])
    )


def _new_rail(axis, feature):
    """Create a new rail initialized with one feature."""
    rail = {
        "axis": axis,
        "coord": feature["coord"],
        "min_coord": feature["coord"],
        "max_coord": feature["coord"],
        "members": [],
        "lines_by_z": defaultdict(list),
        "spans_by_z": defaultdict(list),
    }
    _add_to_rail(rail, feature)
    return rail


def _build_axis_rails(features, axis, merge_tol, eps):
    """Group same-axis line features into shared rails and snap rules."""
    if not features:
        return [], []

    rails = []
    active_start = 0
    sorted_features = sorted(
        features,
        key=lambda item: (
            item["coord"],
            item["z_bottom"],
            item["z_top"],
            item["span_min"],
        ),
    )

    for feature in sorted_features:
        while (
            active_start < len(rails)
            and feature["coord"] - rails[active_start]["max_coord"]
            > merge_tol + eps
        ):
            active_start += 1

        best_index = None
        best_distance = None
        for index in range(active_start, len(rails)):
            rail = rails[index]
            if feature["coord"] < rail["min_coord"] - merge_tol - eps:
                continue
            if not _can_add_to_rail(
                rail,
                feature,
                merge_tol,
                eps,
                features,
            ):
                continue

            distance = abs(feature["coord"] - rail["coord"])
            if best_distance is None or distance < best_distance:
                best_index = index
                best_distance = distance

        if best_index is None:
            rails.append(_new_rail(axis, feature))
        else:
            _add_to_rail(rails[best_index], feature)

    rails.sort(key=lambda rail: rail["coord"])
    public_rails = []
    snap_rules = []

    for rail_id, rail in enumerate(rails):
        public_rails.append(_serialize_rail(rail, rail_id))
        snap_rules.extend(_rail_snap_rules(rail, rail_id))

    snap_rules.sort(
        key=lambda rule: (
            rule["z"],
            rule["rail_id"],
            rule["span_min"],
            rule["span_max"],
        )
    )
    return public_rails, snap_rules


def _serialize_rail(rail, rail_id):
    """Convert an internal rail object into a public rail dictionary."""
    return {
        "axis": rail["axis"],
        "rail_id": rail_id,
        "coord": float(rail["coord"]),
        "min_coord": float(rail["min_coord"]),
        "max_coord": float(rail["max_coord"]),
        "member_count": len(rail["members"]),
        "lines_by_z": {
            z: lines
            for z, lines in sorted(
                rail["lines_by_z"].items(),
                key=lambda item: item[0],
            )
        },
    }


def _rail_snap_rules(rail, rail_id):
    """Create deduplicated snap rules for all features in one rail."""
    snap_rules = []
    seen_rules = set()

    for feature in rail["members"]:
        rule_key = (
            feature["z"],
            round(feature["span_min"], 8),
            round(feature["span_max"], 8),
            round(feature["coord"], 8),
        )
        if rule_key in seen_rules:
            continue

        seen_rules.add(rule_key)
        snap_rules.append(
            {
                "axis": rail["axis"],
                "rail_id": rail_id,
                "rail_coord": float(rail["coord"]),
                "target_coord": float(feature["coord"]),
                "z": feature["z"],
                "z_bottom": feature["z_bottom"],
                "z_top": feature["z_top"],
                "span_min": float(feature["span_min"]),
                "span_max": float(feature["span_max"]),
                "feature_id": feature["feature_id"],
            }
        )

    return snap_rules


def build_shared_rails(lines, merge_tol, eps=0.01, z_decimals=4):
    """Build shared x/y rails and z-indexed snap rules from pattern lines."""
    vertical_lines, horizontal_lines = _classify_line(lines, eps=eps)

    vertical_features = [
        _line_to_feature(
            line,
            "x",
            feature_id,
            eps=eps,
            z_decimals=z_decimals,
        )
        for feature_id, line in enumerate(vertical_lines)
    ]
    horizontal_features = [
        _line_to_feature(
            line,
            "y",
            feature_id,
            eps=eps,
            z_decimals=z_decimals,
        )
        for feature_id, line in enumerate(horizontal_lines)
    ]

    x_rails, x_snap_rules = _build_axis_rails(
        vertical_features,
        "x",
        merge_tol,
        eps,
    )
    y_rails, y_snap_rules = _build_axis_rails(
        horizontal_features,
        "y",
        merge_tol,
        eps,
    )

    snap_rules_by_z = defaultdict(list)
    for rule in x_snap_rules + y_snap_rules:
        snap_rules_by_z[rule["z"]].append(rule)

    snap_rules_by_z = {
        z: sorted(
            rules,
            key=lambda rule: (
                rule["axis"],
                rule["rail_id"],
                rule["span_min"],
            ),
        )
        for z, rules in sorted(
            snap_rules_by_z.items(),
            key=lambda item: item[0],
        )
    }

    return {
        "x_rails": x_rails,
        "y_rails": y_rails,
        "x_list": [rail["coord"] for rail in x_rails],
        "y_list": [rail["coord"] for rail in y_rails],
        "snap_rules_by_z": snap_rules_by_z,
    }
