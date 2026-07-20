"""Build shared checkerboard rails and per-layer snap rules."""

from bisect import bisect_left, bisect_right
from collections import defaultdict
import math

from optimal_checkerboard.algorithms.classify_line import (
    _classify_line,
    _line_components,
)


def _coords_equal(left, right):
    """Return whether two coordinates are the same topology target.

    ``eps`` is deliberately not used here.  It is a geometric search
    tolerance, not permission to replace one pattern coordinate with another.
    """
    return float(left) == float(right)


def _line_to_feature(line, axis, feature_id, eps=None):
    """Convert one geometric line into a normalized feature dictionary."""
    x1, y1, x2, y2, z_bottom, z_top = _line_components(line, eps=eps)

    if axis == "x":
        coord = x1
        span_min = min(y1, y2)
        span_max = max(y1, y2)
    elif axis == "y":
        coord = y1
        span_min = min(x1, x2)
        span_max = max(x1, x2)
    else:
        raise ValueError("axis must be 'x' or 'y'")

    z_bottom = float(z_bottom)
    z_top = float(z_top)
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


def _features_from_lines(lines, axis, eps):
    """Return normalized features with input-order-independent identifiers."""
    features = [
        _line_to_feature(
            line,
            axis,
            feature_id=-1,
            eps=eps,
        )
        for line in lines
    ]
    features.sort(key=_feature_sort_key)
    for feature_id, feature in enumerate(features):
        feature["feature_id"] = feature_id
    return features


def _feature_sort_key(feature):
    """Return the canonical ordering key used by grouping and rule output."""
    return (
        feature["coord"],
        feature["z_bottom"],
        feature["z_top"],
        feature["span_min"],
        feature["span_max"],
        tuple(feature["line"][0]),
        tuple(feature["line"][1]),
    )


def _spans_overlap_or_touch(a_min, a_max, b_min, b_max, eps):
    """Return whether two one-dimensional spans overlap or touch."""
    return max(a_min, b_min) <= min(a_max, b_max)


def _z_ranges_active_overlap(feature, other, eps):
    """Return whether two features are active over the same z interval.

    Ranges conflict when they overlap or share a top/bottom event. This keeps
    the top plane inclusive for positive-height and zero-height features.
    """
    a_min = feature["z_bottom"]
    a_max = feature["z_top"]
    b_min = other["z_bottom"]
    b_max = other["z_top"]
    return max(a_min, b_min) <= min(a_max, b_max)


def _can_add_to_rail(
    rail,
    feature,
    merge_tol,
    eps,
    all_features,
    opposite_pinned_coords=(),
    feature_coords=(),
):
    """Return whether a feature is compatible with an existing shared rail."""
    new_min = min(rail["min_coord"], feature["coord"])
    new_max = max(rail["max_coord"], feature["coord"])
    if new_max - new_min > merge_tol:
        return False

    # A pinned coordinate is a structural boundary.  A rail may contain any
    # number of features at that exact target, but it must never displace it.
    if rail.get("pinned", False) and not _coords_equal(
        feature["coord"], rail["coord"]
    ):
        return False
    if feature.get("pinned", False) and not (
        _coords_equal(rail["min_coord"], feature["coord"])
        and _coords_equal(rail["max_coord"], feature["coord"])
    ):
        return False

    # Pairwise lifecycle/span incompatibilities are precomputed once as
    # Python-int bitsets.  This avoids rescanning every rail member for every
    # greedy candidate at geometry counts near the supported 1,000 limit.
    feature_bit = 1 << feature["feature_id"]
    if rail["_incompatible_mask"] & feature_bit:
        return False

    if _has_blocking_intermediate_feature(
        rail,
        feature,
        new_min,
        new_max,
        all_features,
        eps,
        feature_coords=feature_coords,
    ):
        return False

    return True


def _preserves_adjacent_rail_order(
    rails,
    rail_index,
    feature,
    eps,
    pinned_coords=(),
):
    """Return whether adding a feature keeps neighboring rail targets ordered."""
    candidate_rail = _rail_with_feature(rails[rail_index], feature)

    # Features arrive in canonical coordinate order and accepted rails remain
    # sorted.  A candidate whose member range crossed an existing neighbor
    # could not be valid on either side of that neighbor, so only the current
    # adjacent rails need checking; copying and sorting every rail candidate
    # caused cubic preprocessing on 1,000-feature adversarial inputs.
    if rail_index > 0 and not _adjacent_rails_keep_order(
        rails[rail_index - 1],
        candidate_rail,
        eps,
    ):
        return False
    if (
        rail_index + 1 < len(rails)
        and not _adjacent_rails_keep_order(
            candidate_rail,
            rails[rail_index + 1],
            eps,
        )
    ):
        return False

    pinned_coords = tuple(pinned_coords)
    if pinned_coords:
        insertion = bisect_left(pinned_coords, candidate_rail["coord"])
        if insertion and not _adjacent_rails_keep_order(
            _virtual_pinned_rail(
                candidate_rail["axis"],
                pinned_coords[insertion - 1],
            ),
            candidate_rail,
            eps,
        ):
            return False

        if (
            insertion < len(pinned_coords)
            and pinned_coords[insertion] == candidate_rail["coord"]
        ):
            if not candidate_rail.get("pinned", False):
                return False
            insertion += 1
        if insertion < len(pinned_coords) and not _adjacent_rails_keep_order(
            candidate_rail,
            _virtual_pinned_rail(
                candidate_rail["axis"],
                pinned_coords[insertion],
            ),
            eps,
        ):
            return False
    return True


def _rail_with_feature(rail, feature):
    """Return a lightweight rail snapshot with the feature added."""
    new_min = min(rail["min_coord"], feature["coord"])
    new_max = max(rail["max_coord"], feature["coord"])
    return {
        "axis": rail["axis"],
        "coord": new_min + 0.5 * (new_max - new_min),
        "min_coord": new_min,
        "max_coord": new_max,
        "members": rail["members"] + [feature],
        "pinned": rail.get("pinned", False) or feature.get("pinned", False),
    }


def _adjacent_rails_keep_order(left, right, eps):
    """Return whether two adjacent rails keep left targets before right ones."""
    # Rail order is a topology invariant.  Coordinates that merely differ by
    # less than ``eps`` are still distinct and safe as long as the order is
    # strict.  Equality is not safe because it collapses the cells between the
    # rails.
    if left["coord"] >= right["coord"]:
        return False

    for feature in left["members"]:
        if feature["coord"] >= right["coord"]:
            return False

    for feature in right["members"]:
        if left["coord"] >= feature["coord"]:
            return False

    for left_feature in left["members"]:
        for right_feature in right["members"]:
            if not _z_ranges_active_overlap(left_feature, right_feature, eps):
                continue
            if left_feature["coord"] >= right_feature["coord"]:
                return False

    return True


def _rails_keep_global_order(rails, eps, pinned_coords=()):
    """Return whether every rail and virtual pinned boundary stays ordered."""
    simulated_rails = _rails_with_virtual_pins(
        rails,
        pinned_coords=pinned_coords,
    )
    return all(
        _adjacent_rails_keep_order(left, right, eps)
        for left, right in zip(simulated_rails, simulated_rails[1:])
    )


def _rails_with_virtual_pins(rails, pinned_coords=()):
    """Return sorted rails including validation-only pinned boundaries."""
    simulated_rails = list(rails)
    represented_pins = {
        rail["coord"]
        for rail in simulated_rails
        if rail.get("pinned", False)
    }
    axis = simulated_rails[0]["axis"] if simulated_rails else None

    for coord in pinned_coords:
        if coord in represented_pins:
            continue
        simulated_rails.append(_virtual_pinned_rail(axis, coord))

    simulated_rails.sort(key=_rail_sort_key)
    return simulated_rails


def _rail_sort_key(rail):
    """Return a deterministic ordering key for an internal rail."""
    return (
        rail["coord"],
        rail["min_coord"],
        rail["max_coord"],
        not rail.get("pinned", False),
    )


def _virtual_pinned_rail(axis, coord):
    """Create a validation-only rail for a structural boundary coordinate."""
    return {
        "axis": axis,
        "coord": coord,
        "min_coord": coord,
        "max_coord": coord,
        "members": [],
        "pinned": True,
    }


def _has_near_endpoint_conflict(
    rail,
    feature,
    merge_tol,
    eps,
    opposite_pinned_coords=(),
):
    """Return whether near active endpoints should prevent rail sharing."""
    if not isinstance(opposite_pinned_coords, (set, frozenset)):
        opposite_pinned_coords = frozenset(opposite_pinned_coords)
    for member in rail["members"]:
        if not _z_ranges_active_overlap(member, feature, eps):
            continue
        if _coords_equal(member["coord"], feature["coord"]):
            continue

        span_gap = _span_gap(
            member["span_min"],
            member["span_max"],
            feature["span_min"],
            feature["span_max"],
        )
        # Include both the small-gap and merge-tolerance boundaries.  Allowing
        # either boundary lets independent x/y grouping merge two diagonal
        # corners into the same checkerboard node.
        if 0.0 < span_gap <= merge_tol:
            facing_endpoints = _facing_span_endpoints(member, feature)
            if all(
                endpoint in opposite_pinned_coords
                for endpoint in facing_endpoints
            ):
                continue
            return True

    return False


def _facing_span_endpoints(feature, other):
    """Return the two endpoints facing across a positive span gap."""
    if feature["span_max"] < other["span_min"]:
        return feature["span_max"], other["span_min"]
    return other["span_max"], feature["span_min"]


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
    feature_coords=(),
):
    """Return whether an active intermediate feature blocks a rail merge.

    A feature whose coordinate lies between the proposed rail bounds acts as
    an ordering barrier when its active z range and span overlap the new
    feature or an existing rail member.
    """
    if not feature_coords:
        feature_coords = tuple(item["coord"] for item in all_features)
    range_start = bisect_right(feature_coords, new_min)
    range_end = bisect_left(feature_coords, new_max)
    if range_start >= range_end:
        return False

    coordinate_range_mask = (
        ((1 << (range_end - range_start)) - 1) << range_start
    )
    candidate_mask = rail["_member_mask"] | (
        1 << feature["feature_id"]
    )
    overlap_mask = rail["_overlap_mask"] | feature["_overlap_mask"]
    return bool(
        overlap_mask
        & coordinate_range_mask
        & ~candidate_mask
    )


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
    rail["coord"] = rail["min_coord"] + 0.5 * (
        rail["max_coord"] - rail["min_coord"]
    )
    rail["pinned"] = rail.get("pinned", False) or feature.get("pinned", False)
    feature_bit = 1 << feature["feature_id"]
    rail["_member_mask"] |= feature_bit
    rail["_incompatible_mask"] |= feature["_incompatible_mask"]
    rail["_overlap_mask"] |= feature["_overlap_mask"]
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
        "pinned": feature.get("pinned", False),
        "_member_mask": 0,
        "_incompatible_mask": 0,
        "_overlap_mask": 0,
        "lines_by_z": defaultdict(list),
        "spans_by_z": defaultdict(list),
    }
    _add_to_rail(rail, feature)
    return rail


def _prepare_feature_relations(
    features,
    merge_tol,
    eps,
    opposite_pinned_coords,
):
    """Precompute pairwise overlap/conflict relations as compact bitsets."""
    for expected_id, feature in enumerate(features):
        if feature["feature_id"] != expected_id:
            raise RuntimeError("feature ids must follow canonical sort order")
        feature["_overlap_mask"] = 0
        feature["_incompatible_mask"] = 0

    for left_index, left in enumerate(features):
        left_bit = 1 << left_index
        for right_index in range(left_index + 1, len(features)):
            right = features[right_index]
            if not _z_ranges_active_overlap(left, right, eps):
                continue

            spans_touch = _spans_overlap_or_touch(
                left["span_min"],
                left["span_max"],
                right["span_min"],
                right["span_max"],
                eps,
            )
            right_bit = 1 << right_index
            if spans_touch:
                left["_overlap_mask"] |= right_bit
                right["_overlap_mask"] |= left_bit

            incompatible = False
            if not _coords_equal(left["coord"], right["coord"]):
                if spans_touch:
                    incompatible = True
                else:
                    span_gap = _span_gap(
                        left["span_min"],
                        left["span_max"],
                        right["span_min"],
                        right["span_max"],
                    )
                    if 0.0 < span_gap <= merge_tol:
                        facing_endpoints = _facing_span_endpoints(left, right)
                        incompatible = not all(
                            endpoint in opposite_pinned_coords
                            for endpoint in facing_endpoints
                        )

            if incompatible:
                left["_incompatible_mask"] |= right_bit
                right["_incompatible_mask"] |= left_bit


def _build_axis_rails(
    features,
    axis,
    merge_tol,
    eps,
    pinned_coords=(),
    opposite_pinned_coords=(),
):
    """Group same-axis line features into shared rails and snap rules."""
    if not features:
        return [], [], []

    opposite_pinned_coords = frozenset(opposite_pinned_coords)
    _prepare_feature_relations(
        features,
        merge_tol,
        eps,
        opposite_pinned_coords,
    )
    rails = []
    active_start = 0
    sorted_features = sorted(
        features,
        key=_feature_sort_key,
    )
    feature_coords = tuple(
        feature["coord"] for feature in sorted_features
    )

    for feature in sorted_features:
        while (
            active_start < len(rails)
            and feature["coord"] - rails[active_start]["max_coord"]
            > merge_tol
        ):
            active_start += 1

        best_index = None
        best_distance = None
        for index in range(active_start, len(rails)):
            rail = rails[index]
            if feature["coord"] < rail["min_coord"] - merge_tol:
                continue
            if not _can_add_to_rail(
                rail,
                feature,
                merge_tol,
                eps,
                features,
                opposite_pinned_coords=opposite_pinned_coords,
                feature_coords=feature_coords,
            ):
                continue
            if not _preserves_adjacent_rail_order(
                rails,
                index,
                feature,
                eps,
                pinned_coords=pinned_coords,
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

    rails.sort(key=_rail_sort_key)
    if not _rails_keep_global_order(
        rails,
        eps,
        pinned_coords=pinned_coords,
    ):
        # A rail created late in the greedy traversal can invalidate a merge
        # that was safe relative to the rails known at the time.  Exact-target
        # rails are the conservative, deterministic fallback and never alter
        # a pattern coordinate.
        rails = _build_exact_coordinate_rails(features, axis)
        if not _rails_keep_global_order(
            rails,
            eps,
            pinned_coords=pinned_coords,
        ):
            raise RuntimeError("Unable to preserve global checkerboard rail order")

    public_rails = []
    snap_rules = []
    restore_rules = []

    for rail_id, rail in enumerate(rails):
        public_rails.append(_serialize_rail(rail, rail_id))
        rail_snap_rules, rail_restore_rules = _rail_rules(rail, rail_id, eps)
        snap_rules.extend(rail_snap_rules)
        restore_rules.extend(rail_restore_rules)

    snap_rules.sort(
        key=lambda rule: (
            rule["z"],
            rule["rail_id"],
            rule["span_min"],
            rule["span_max"],
        )
    )
    restore_rules.sort(
        key=lambda rule: (
            rule["z"],
            rule["rail_id"],
            rule["span_min"],
            rule["span_max"],
        )
    )
    return public_rails, snap_rules, restore_rules


def _build_exact_coordinate_rails(features, axis):
    """Build the no-displacement fallback grouped only by exact target."""
    rails_by_coord = {}
    for feature in sorted(
        features,
        key=_feature_sort_key,
    ):
        rail = rails_by_coord.get(feature["coord"])
        if rail is None:
            rail = _new_rail(axis, feature)
            rails_by_coord[feature["coord"]] = rail
        else:
            _add_to_rail(rail, feature)
    return sorted(rails_by_coord.values(), key=_rail_sort_key)


def _serialize_rail(rail, rail_id):
    """Convert an internal rail object into a public rail dictionary."""
    return {
        "axis": rail["axis"],
        "rail_id": rail_id,
        "coord": float(rail["coord"]),
        "min_coord": float(rail["min_coord"]),
        "max_coord": float(rail["max_coord"]),
        "member_count": len(rail["members"]),
        "pinned": bool(rail.get("pinned", False)),
        "lines_by_z": {
            z: lines
            for z, lines in sorted(
                rail["lines_by_z"].items(),
                key=lambda item: item[0],
            )
        },
    }


def _same_restore_track(feature, other, eps):
    """Return whether two features share one target and span lifecycle."""
    return (
        _coords_equal(feature["coord"], other["coord"])
        and _coords_equal(feature["span_min"], other["span_min"])
        and _coords_equal(feature["span_max"], other["span_max"])
    )


def _restore_is_deferred(feature, members, eps):
    """Return whether a matching active feature extends beyond this top z."""
    for other in members:
        if other is feature:
            continue
        if other["z_top"] <= feature["z_top"]:
            continue
        if not _same_restore_track(feature, other, eps):
            continue
        if _z_ranges_active_overlap(feature, other, eps):
            return True
    return False


def _rail_rules(rail, rail_id, eps):
    """Create deduplicated snap and restore rules for all rail features."""
    snap_rules = []
    restore_rules = []
    seen_snap_rules = set()
    seen_restore_rules = set()

    for feature in rail["members"]:
        snap_rule_key = (
            feature["z"],
            feature["z_top"],
            feature["span_min"],
            feature["span_max"],
            feature["coord"],
        )
        if snap_rule_key not in seen_snap_rules:
            seen_snap_rules.add(snap_rule_key)
            snap_rules.append(
                {
                    "kind": "snap",
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

        if _restore_is_deferred(feature, rail["members"], eps):
            continue

        restore_rule_key = (
            feature["z_top"],
            feature["span_min"],
            feature["span_max"],
            rail["coord"],
        )
        if restore_rule_key in seen_restore_rules:
            continue

        seen_restore_rules.add(restore_rule_key)
        restore_rules.append(
            {
                "kind": "restore",
                "axis": rail["axis"],
                "rail_id": rail_id,
                "rail_coord": float(rail["coord"]),
                "target_coord": float(rail["coord"]),
                "z": feature["z_top"],
                "z_bottom": feature["z_bottom"],
                "z_top": feature["z_top"],
                "span_min": float(feature["span_min"]),
                "span_max": float(feature["span_max"]),
                "feature_id": feature["feature_id"],
            }
        )

    return snap_rules, restore_rules


def build_shared_rails(
    lines,
    merge_tol,
    eps=None,
    z_decimals=None,
    pinned_coords=None,
):
    """Build shared x/y rails and z-indexed snap rules from pattern lines.

    ``eps`` only caps scale-aware input-noise handling during line
    classification.  Rail targets, spans, z events, and merge bounds retain
    their exact topology values.  ``z_decimals`` remains accepted for source
    compatibility but z events are never rounded.
    """
    merge_tol = float(merge_tol)
    if not math.isfinite(merge_tol) or merge_tol < 0.0:
        raise ValueError("merge_tol must be a non-negative finite number")

    pinned_coords = _normalize_pinned_coords(pinned_coords)
    vertical_lines, horizontal_lines = _classify_line(lines, eps=eps)

    vertical_features = _features_from_lines(
        vertical_lines,
        "x",
        eps,
    )
    horizontal_features = _features_from_lines(
        horizontal_lines,
        "y",
        eps,
    )

    _mark_pinned_features(vertical_features, pinned_coords["x"])
    _mark_pinned_features(horizontal_features, pinned_coords["y"])

    x_rails, x_snap_rules, x_restore_rules = _build_axis_rails(
        vertical_features,
        "x",
        merge_tol,
        eps,
        pinned_coords=pinned_coords["x"],
        opposite_pinned_coords=pinned_coords["y"],
    )
    y_rails, y_snap_rules, y_restore_rules = _build_axis_rails(
        horizontal_features,
        "y",
        merge_tol,
        eps,
        pinned_coords=pinned_coords["y"],
        opposite_pinned_coords=pinned_coords["x"],
    )

    snap_rules_by_z = defaultdict(list)
    for rule in x_snap_rules + y_snap_rules:
        snap_rules_by_z[rule["z"]].append(rule)

    restore_rules_by_z = defaultdict(list)
    for rule in x_restore_rules + y_restore_rules:
        restore_rules_by_z[rule["z"]].append(rule)

    return {
        "x_rails": x_rails,
        "y_rails": y_rails,
        "x_list": [rail["coord"] for rail in x_rails],
        "y_list": [rail["coord"] for rail in y_rails],
        "snap_rules_by_z": _rules_by_z(snap_rules_by_z),
        "restore_rules_by_z": _rules_by_z(restore_rules_by_z),
    }


def _normalize_pinned_coords(pinned_coords):
    """Validate optional structural coordinates for each mesh axis."""
    if pinned_coords is None:
        return {"x": (), "y": ()}
    if not isinstance(pinned_coords, dict):
        raise TypeError("pinned_coords must be a mapping with x/y entries")

    unknown_axes = set(pinned_coords) - {"x", "y"}
    if unknown_axes:
        raise ValueError("pinned_coords only supports the x and y axes")

    normalized = {}
    for axis in ("x", "y"):
        values = []
        for raw_value in pinned_coords.get(axis, ()):
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError("pinned coordinates must be finite")
            values.append(value)
        normalized[axis] = tuple(sorted(set(values)))
    return normalized


def _mark_pinned_features(features, pinned_coords):
    """Attach structural-boundary provenance to exact matching features."""
    pinned_coord_set = set(pinned_coords)
    for feature in features:
        feature["pinned"] = feature["coord"] in pinned_coord_set


def _rules_by_z(rules_by_z):
    """Return z-indexed rules with stable ordering inside each z bucket."""
    return {
        z: sorted(
            rules,
            key=lambda rule: (
                rule["axis"],
                rule["rail_id"],
                rule["span_min"],
            ),
        )
        for z, rules in sorted(
            rules_by_z.items(),
            key=lambda item: item[0],
        )
    }
