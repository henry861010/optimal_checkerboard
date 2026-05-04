"""Build shared checkerboard rails and per-layer snap rules."""

from collections import defaultdict

from optimal_checkerboard.algorithms.classify_line import _classify_line


def _round_key(value, decimals=4):
    """Round a coordinate into a stable dictionary key."""
    return round(float(value), decimals)


def _line_to_feature(line, axis, feature_id, z_decimals=4):
    """Convert one geometric line into a normalized feature dictionary."""
    (x1, y1, z1), (x2, y2, z2) = line

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

    z = _round_key((z1 + z2) / 2.0, z_decimals)
    return {
        "feature_id": feature_id,
        "axis": axis,
        "coord": float(coord),
        "z": z,
        "span_min": float(span_min),
        "span_max": float(span_max),
        "line": [
            [float(x1), float(y1), float(z1)],
            [float(x2), float(y2), float(z2)],
        ],
    }


def _spans_overlap_or_touch(a_min, a_max, b_min, b_max, eps):
    """Return whether two one-dimensional spans overlap or touch."""
    return max(a_min, b_min) <= min(a_max, b_max) + eps


def _can_add_to_rail(rail, feature, merge_tol, eps, features_by_z):
    """Return whether a feature is compatible with an existing shared rail."""
    new_min = min(rail["min_coord"], feature["coord"])
    new_max = max(rail["max_coord"], feature["coord"])
    if new_max - new_min > merge_tol + eps:
        return False

    # Same-z overlapping spans cannot share a rail if they snap to different
    # target coordinates, because the same checkerboard node would have two
    # destinations at the same drag event.
    for span_min, span_max, coord in rail["spans_by_z"].get(feature["z"], []):
        if (
            _spans_overlap_or_touch(
                span_min,
                span_max,
                feature["span_min"],
                feature["span_max"],
                eps,
            )
            and abs(coord - feature["coord"]) > eps
        ):
            return False

    if _has_blocking_intermediate_feature(
        rail,
        feature,
        new_min,
        new_max,
        features_by_z,
        eps,
    ):
        return False

    return True


def _has_blocking_intermediate_feature(
    rail,
    feature,
    new_min,
    new_max,
    features_by_z,
    eps,
):
    """Return whether a same-z feature blocks a proposed rail merge.

    A same-z feature whose coordinate lies between the proposed rail bounds
    acts as an ordering barrier if its span overlaps the new feature or an
    existing rail member.  This prevents merging x=10 and x=12 through an
    intervening conflicting x=11 feature.
    """
    member_ids = {member["feature_id"] for member in rail["members"]}
    candidate_ids = member_ids | {feature["feature_id"]}
    same_z_features = features_by_z.get(feature["z"], [])
    rail_same_z_members = [
        member for member in rail["members"] if member["z"] == feature["z"]
    ]

    for other in same_z_features:
        if other["feature_id"] in candidate_ids:
            continue
        if not new_min + eps < other["coord"] < new_max - eps:
            continue
        if _feature_overlaps_any(other, [feature] + rail_same_z_members, eps):
            return True

    return False


def _feature_overlaps_any(feature, others, eps):
    """Return whether a feature span overlaps any feature in a list."""
    for other in others:
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

    features_by_z = defaultdict(list)
    for feature in features:
        features_by_z[feature["z"]].append(feature)

    rails = []
    active_start = 0
    sorted_features = sorted(
        features,
        key=lambda item: (item["coord"], item["z"], item["span_min"]),
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
                features_by_z,
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
        _line_to_feature(line, "x", feature_id, z_decimals=z_decimals)
        for feature_id, line in enumerate(vertical_lines)
    ]
    horizontal_features = [
        _line_to_feature(line, "y", feature_id, z_decimals=z_decimals)
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
