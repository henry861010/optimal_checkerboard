"""Geometry-scale safety preflight for shared checkerboard snap plans.

The validator models the structured two-dimensional grid without allocating
its node or element arrays.  Only nodes touched by active snap rules and the
cells incident to those nodes are materialized.  This keeps the work tied to
the one-dimensional axes, geometry rules, and affected cells instead of the
eventual 2.5D mesh size.
"""

from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
import math
from typing import Optional


@dataclass(frozen=True)
class SnapPlanZState:
    """One exact z boundary or one open interval between boundaries."""

    kind: str
    value: Optional[float] = None
    lower: Optional[float] = None
    upper: Optional[float] = None

    def describe(self):
        """Return a compact human-readable description."""
        if self.kind == "boundary":
            return f"z={self.value!r}"
        return f"z in ({self.lower!r}, {self.upper!r})"


@dataclass(frozen=True)
class SnapPlanFailure:
    """Concrete evidence for the first unsafe snap-plan state."""

    reason: str
    state: SnapPlanZState
    detail: str
    axis: Optional[str] = None
    rail_id: Optional[object] = None
    node: Optional[tuple[int, int]] = None
    cell: Optional[tuple[int, int]] = None
    vertices: Optional[tuple[tuple[float, float], ...]] = None
    cross_products: Optional[tuple[float, ...]] = None
    rule_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class SnapPlanValidationResult:
    """Result and bounded-work counters for one snap-plan preflight."""

    safe: bool
    critical_state_count: int
    evaluated_state_count: int
    checked_cell_count: int
    failure: Optional[SnapPlanFailure] = None
    enumerated_node_count: int = 0


class UnsafeSnapPlanError(ValueError):
    """Raised when a shared-rail plan can damage structured 2D topology."""

    def __init__(self, failure):
        self.failure = failure
        super().__init__(
            "Unsafe shared snap plan at "
            f"{failure.state.describe()}: {failure.detail}"
        )


@dataclass(frozen=True)
class _Rule:
    index: int
    axis: str
    rail_id: object
    rail_coord: float
    rail_axis_index: int
    target: float
    span_min: float
    span_max: float
    z_bottom: float
    z_top: float


def validate_snap_plan(
    axis_coordinates,
    rails_by_axis,
    snap_rules,
    *,
    raise_on_unsafe=True,
):
    """Validate every topologically distinct z state of a shared snap plan.

    Args:
        axis_coordinates: Full structural node stations as
            ``{"x": x_nodes, "y": y_nodes}``.  These must include rail
            defaults, pinned coordinates, span endpoints, and filler stations.
        rails_by_axis: Serialized rails as ``{"x": [...], "y": [...]}``.
        snap_rules: Either a z-to-rules mapping or one iterable of rules.
        raise_on_unsafe: Raise :class:`UnsafeSnapPlanError` on the first
            unsafe state.  When false, return a result containing ``failure``.

    The active-state model treats both z ends as inclusive, matching pattern
    feature lifecycles.  Every exact boundary and every open interval between
    adjacent boundaries is considered.  Repeated active-rule signatures are
    evaluated once because they produce identical two-dimensional geometry.
    """
    axes = _normalize_axes(axis_coordinates)
    rail_indexes = _normalize_rails(rails_by_axis, axes)
    rules = _normalize_rules(snap_rules, rail_indexes)
    states = _critical_states(rules)

    checked_cell_count = 0
    enumerated_node_count = 0
    evaluated_state_count = 0
    safe_signatures = set()

    for state in states:
        active = tuple(
            rule for rule in rules if _rule_active_in_state(rule, state)
        )
        signature = frozenset(_rule_geometry_key(rule) for rule in active)
        if signature in safe_signatures:
            continue

        evaluated_state_count += 1
        failure, checked, enumerated = _validate_active_state(
            axes,
            active,
            state,
        )
        checked_cell_count += checked
        enumerated_node_count += enumerated
        if failure is not None:
            result = SnapPlanValidationResult(
                safe=False,
                critical_state_count=len(states),
                evaluated_state_count=evaluated_state_count,
                checked_cell_count=checked_cell_count,
                failure=failure,
                enumerated_node_count=enumerated_node_count,
            )
            if raise_on_unsafe:
                raise UnsafeSnapPlanError(failure)
            return result
        safe_signatures.add(signature)

    return SnapPlanValidationResult(
        safe=True,
        critical_state_count=len(states),
        evaluated_state_count=evaluated_state_count,
        checked_cell_count=checked_cell_count,
        enumerated_node_count=enumerated_node_count,
    )


def preflight_snap_plan(*args, **kwargs):
    """Alias emphasizing that validation happens before mesh allocation."""
    return validate_snap_plan(*args, **kwargs)


def _normalize_axes(axis_coordinates):
    if not isinstance(axis_coordinates, Mapping):
        raise TypeError("axis_coordinates must be a mapping with x/y entries")

    axes = {}
    for axis in ("x", "y"):
        try:
            raw_values = axis_coordinates[axis]
        except KeyError as exc:
            raise ValueError(
                "axis_coordinates must contain both x and y"
            ) from exc
        try:
            values = tuple(float(value) for value in raw_values)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"axis_coordinates[{axis!r}] must contain numeric values"
            ) from exc
        if len(values) < 2:
            raise ValueError(
                f"axis_coordinates[{axis!r}] must contain at least two values"
            )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("structural axis coordinates must be finite")
        if any(left >= right for left, right in zip(values, values[1:])):
            raise ValueError(
                "structural axis coordinates must be strictly increasing"
            )
        axes[axis] = values
    return axes


def _normalize_rails(rails_by_axis, axes):
    if not isinstance(rails_by_axis, Mapping):
        raise TypeError("rails_by_axis must be a mapping with x/y entries")

    result = {}
    for axis in ("x", "y"):
        raw_rails = rails_by_axis.get(axis, ())
        by_id = {}
        by_coord = {}
        coord_to_axis_index = {
            coord: index for index, coord in enumerate(axes[axis])
        }
        for fallback_id, rail in enumerate(raw_rails):
            if not isinstance(rail, Mapping):
                raise TypeError("each rail must be a mapping")
            rail_axis = rail.get("axis", axis)
            if rail_axis != axis:
                raise ValueError("rail axis does not match its axis bucket")
            rail_id = rail.get("rail_id", fallback_id)
            if rail_id in by_id:
                raise ValueError(f"duplicate {axis}-axis rail_id {rail_id!r}")
            coord = _finite_float(rail.get("coord"), "rail coord")
            if coord not in coord_to_axis_index:
                raise ValueError(
                    f"{axis}-axis rail {rail_id!r} coord {coord!r} is absent "
                    "from structural axis coordinates"
                )
            if coord in by_coord:
                raise ValueError(
                    f"multiple {axis}-axis rails use coord {coord!r}"
                )
            normalized = {
                "rail_id": rail_id,
                "coord": coord,
                "axis_index": coord_to_axis_index[coord],
                "pinned": bool(rail.get("pinned", False)),
            }
            by_id[rail_id] = normalized
            by_coord[coord] = normalized
        result[axis] = {"by_id": by_id, "by_coord": by_coord}
    return result


def _normalize_rules(snap_rules, rail_indexes):
    flattened = _flatten_rules(snap_rules)
    normalized = []
    for index, (bucket_z, raw_rule) in enumerate(flattened):
        if not isinstance(raw_rule, Mapping):
            raise TypeError("each snap rule must be a mapping")
        axis = raw_rule.get("axis")
        if axis not in ("x", "y"):
            raise ValueError("snap rule axis must be 'x' or 'y'")

        rail = _resolve_rule_rail(raw_rule, rail_indexes[axis], axis)
        rail_coord = _finite_float(
            raw_rule.get("rail_coord", rail["coord"]),
            "snap rule rail_coord",
        )
        if rail_coord != rail["coord"]:
            raise ValueError(
                f"snap rule rail_coord does not match {axis}-axis rail "
                f"{rail['rail_id']!r}"
            )
        target = _finite_float(
            raw_rule.get("target_coord"),
            "snap rule target_coord",
        )
        span_min = _finite_float(raw_rule.get("span_min"), "rule span_min")
        span_max = _finite_float(raw_rule.get("span_max"), "rule span_max")
        if span_min > span_max:
            raise ValueError("snap rule span_min must not exceed span_max")

        raw_bottom = raw_rule.get("z_bottom", raw_rule.get("z", bucket_z))
        if raw_bottom is None:
            raise ValueError("snap rule must provide z or z_bottom")
        z_bottom = _finite_float(raw_bottom, "snap rule z_bottom")
        z_top = _finite_float(
            raw_rule.get("z_top", z_bottom),
            "snap rule z_top",
        )
        if z_bottom > z_top:
            raise ValueError("snap rule z_bottom must not exceed z_top")

        if rail["pinned"] and target != rail_coord:
            raise ValueError(
                f"pinned {axis}-axis rail {rail['rail_id']!r} has a moving "
                "snap target"
            )

        normalized.append(
            _Rule(
                index=index,
                axis=axis,
                rail_id=rail["rail_id"],
                rail_coord=rail_coord,
                rail_axis_index=rail["axis_index"],
                target=target,
                span_min=span_min,
                span_max=span_max,
                z_bottom=z_bottom,
                z_top=z_top,
            )
        )
    return tuple(normalized)


def _flatten_rules(snap_rules):
    if snap_rules is None:
        return []
    if isinstance(snap_rules, Mapping):
        flattened = []
        for bucket_z, rules in snap_rules.items():
            if isinstance(rules, Mapping) or not isinstance(rules, Iterable):
                raise TypeError("each snap-rule bucket must be an iterable")
            flattened.extend((bucket_z, rule) for rule in rules)
        return flattened
    if isinstance(snap_rules, (str, bytes)) or not isinstance(
        snap_rules, Iterable
    ):
        raise TypeError("snap_rules must be a mapping or iterable")
    return [(None, rule) for rule in snap_rules]


def _resolve_rule_rail(rule, rail_index, axis):
    if "rail_id" in rule:
        rail_id = rule["rail_id"]
        try:
            return rail_index["by_id"][rail_id]
        except KeyError as exc:
            raise ValueError(
                f"snap rule references unknown {axis}-axis rail_id "
                f"{rail_id!r}"
            ) from exc
    if "rail_coord" in rule:
        coord = _finite_float(rule["rail_coord"], "snap rule rail_coord")
        try:
            return rail_index["by_coord"][coord]
        except KeyError as exc:
            raise ValueError(
                f"snap rule references unknown {axis}-axis rail coord "
                f"{coord!r}"
            ) from exc
    raise ValueError("snap rule must provide rail_id or rail_coord")


def _finite_float(value, name):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _critical_states(rules):
    boundaries = sorted(
        {value for rule in rules for value in (rule.z_bottom, rule.z_top)}
    )
    states = []
    for index, boundary in enumerate(boundaries):
        states.append(SnapPlanZState(kind="boundary", value=boundary))
        if index + 1 < len(boundaries):
            states.append(
                SnapPlanZState(
                    kind="interval",
                    lower=boundary,
                    upper=boundaries[index + 1],
                )
            )
    return tuple(states)


def _rule_active_in_state(rule, state):
    if state.kind == "boundary":
        return rule.z_bottom <= state.value <= rule.z_top
    return rule.z_bottom <= state.lower and rule.z_top >= state.upper


def _rule_geometry_key(rule):
    """Return lifecycle-independent geometry for one active rule."""
    return (
        rule.axis,
        rule.rail_id,
        rule.rail_coord,
        rule.rail_axis_index,
        rule.target,
        rule.span_min,
        rule.span_max,
    )


def _validate_active_state(axes, active, state):
    nx = len(axes["x"])
    ny = len(axes["y"])
    active_by_axis = {
        "x": [rule for rule in active if rule.axis == "x"],
        "y": [rule for rule in active if rule.axis == "y"],
    }
    span_bounds = {}
    updates = {"x": {}, "y": {}}
    writers = {"x": {}, "y": {}}
    enumerated_node_count = 0

    moving_intervals = {}
    for rule in active:
        span_axis = "y" if rule.axis == "x" else "x"
        span_values = axes[span_axis]
        lo = bisect_left(span_values, rule.span_min)
        hi = bisect_right(span_values, rule.span_max)
        span_bounds[rule.index] = (lo, hi)

        # A no-op rule does not write any runtime coordinate.  Retain its
        # compact span bounds for representation checks, but do not expand a
        # potentially full-axis range into Python node objects.
        if rule.target == rule.rail_coord:
            continue

        moving_intervals.setdefault(
            (rule.axis, rule.rail_axis_index, rule.target),
            [],
        ).append((lo, hi, rule))

    # Expand the union of same-target spans once. Different targets remain in
    # separate groups, so _assign_update still detects every shared-node
    # conflict exactly.
    for intervals in moving_intervals.values():
        for lo, hi, rule in _merged_rule_intervals(intervals):
            for span_index in range(lo, hi):
                if rule.axis == "x":
                    node_id = span_index * nx + rule.rail_axis_index
                else:
                    node_id = rule.rail_axis_index * nx + span_index
                enumerated_node_count += 1
                failure = _assign_update(
                    updates,
                    writers,
                    rule,
                    node_id,
                    nx,
                    state,
                )
                if failure is not None:
                    return (
                        _with_complete_rule_diagnostics(
                            failure,
                            active_by_axis,
                            axes,
                            nx,
                        ),
                        0,
                        enumerated_node_count,
                    )

    # Match the runtime's coupled-corner rule: perpendicular feature targets
    # that cross each other's spans also move their structural rail
    # intersection, even when that default intersection is outside a span.
    for x_rule, y_rule in _iter_coupled_pairs(active_by_axis):
        enumerated_node_count += 1
        node_id = y_rule.rail_axis_index * nx + x_rule.rail_axis_index
        for rule in (x_rule, y_rule):
            failure = _assign_update(
                updates,
                writers,
                rule,
                node_id,
                nx,
                state,
            )
            if failure is not None:
                return (
                    _with_complete_rule_diagnostics(
                        failure,
                        active_by_axis,
                        axes,
                        nx,
                    ),
                    0,
                    enumerated_node_count,
                )

    affected_cells = _affected_cells(updates, nx, ny)
    for cell_id in affected_cells:
        cell_y, cell_x = divmod(cell_id, nx - 1)
        node_ids = (
            cell_y * nx + cell_x,
            cell_y * nx + cell_x + 1,
            (cell_y + 1) * nx + cell_x + 1,
            (cell_y + 1) * nx + cell_x,
        )
        vertices = tuple(
            _node_coordinate(
                node_id,
                axes,
                updates,
                nx,
            )
            for node_id in node_ids
        )
        crosses = _convex_cross_products(vertices)
        if any(cross <= 0.0 for cross in crosses):
            related_rules = set()
            for axis in ("x", "y"):
                for node_id in node_ids:
                    writer = writers[axis].get(node_id)
                    if writer is not None:
                        related_rules.add(writer)
            failure = SnapPlanFailure(
                reason="non_ccw_or_non_convex_cell",
                state=state,
                detail=(
                    "snap targets collapse, invert, or make concave "
                    f"structured cell ({cell_x}, {cell_y})"
                ),
                cell=(cell_x, cell_y),
                vertices=vertices,
                cross_products=crosses,
                rule_indices=tuple(sorted(related_rules)),
            )
            return (
                _with_complete_rule_diagnostics(
                    failure,
                    active_by_axis,
                    axes,
                    nx,
                ),
                len(affected_cells),
                enumerated_node_count,
            )

    representation_failure = _validate_feature_representation(
        axes,
        active,
        active_by_axis,
        span_bounds,
        updates,
        nx,
        state,
    )
    return (
        representation_failure,
        len(affected_cells),
        enumerated_node_count,
    )


def _merged_rule_intervals(intervals):
    """Yield discrete unions for one axis/rail/target update group."""
    ordered = sorted(intervals, key=lambda item: (item[0], item[1]))
    if not ordered:
        return
    current_lo, current_hi, representative = ordered[0]
    for lo, hi, rule in ordered[1:]:
        if lo <= current_hi:
            current_hi = max(current_hi, hi)
            continue
        yield current_lo, current_hi, representative
        current_lo, current_hi, representative = lo, hi, rule
    yield current_lo, current_hi, representative


def _unique_geometry_rules(rules):
    """Keep one provenance representative per effective geometry rule."""
    unique = {}
    for rule in rules:
        unique.setdefault(_rule_geometry_key(rule), rule)
    return tuple(unique.values())


def _iter_coupled_pairs(active_by_axis):
    """Yield target-crossing perpendicular pairs that can change geometry."""
    y_rules_by_target = sorted(
        _unique_geometry_rules(active_by_axis["y"]),
        key=lambda rule: rule.target,
    )
    y_targets = [rule.target for rule in y_rules_by_target]
    for x_rule in _unique_geometry_rules(active_by_axis["x"]):
        lo = bisect_left(y_targets, x_rule.span_min)
        hi = bisect_right(y_targets, x_rule.span_max)
        for index in range(lo, hi):
            y_rule = y_rules_by_target[index]
            if not y_rule.span_min <= x_rule.target <= y_rule.span_max:
                continue
            # With two no-op rules, their intersection is already present on
            # both structural rail spans and neither runtime coordinate moves.
            if (
                x_rule.target == x_rule.rail_coord
                and y_rule.target == y_rule.rail_coord
            ):
                continue
            yield x_rule, y_rule


def _assign_update(updates, writers, rule, node_id, nx, state):
    # Runtime snap application skips no-op coordinates but still uses the rule
    # for perpendicular coupled-corner discovery.
    if rule.target == rule.rail_coord:
        return None

    axis_updates = updates[rule.axis]
    previous = axis_updates.get(node_id)
    if previous is not None and previous != rule.target:
        node_y, node_x = divmod(node_id, nx)
        previous_writer = writers[rule.axis].get(node_id)
        related_rules = {rule.index}
        if previous_writer is not None:
            related_rules.add(previous_writer)
        return SnapPlanFailure(
            reason="conflicting_node_targets",
            state=state,
            detail=(
                f"{rule.axis}-axis structural node ({node_x}, {node_y}) "
                f"receives both {previous!r} and {rule.target!r}"
            ),
            axis=rule.axis,
            rail_id=rule.rail_id,
            node=(node_x, node_y),
            rule_indices=tuple(sorted(related_rules)),
        )
    axis_updates[node_id] = rule.target
    # One concrete writer is sufficient for both conflict and invalid-cell
    # diagnostics.  Keeping every same-target writer would recreate the
    # rule-count x span-length Python-object blow-up this preflight avoids.
    writers[rule.axis].setdefault(node_id, rule.index)
    return None


def _with_complete_rule_diagnostics(failure, active_by_axis, axes, nx):
    """Recover all node writers lazily only on an unsafe state."""
    node_ids = []
    if failure.node is not None:
        node_x, node_y = failure.node
        node_ids.append(node_y * nx + node_x)
    elif failure.cell is not None:
        cell_x, cell_y = failure.cell
        node_ids.extend(
            (
                cell_y * nx + cell_x,
                cell_y * nx + cell_x + 1,
                (cell_y + 1) * nx + cell_x + 1,
                (cell_y + 1) * nx + cell_x,
            )
        )
    else:
        return failure

    related = set(failure.rule_indices)
    axes_to_check = (failure.axis,) if failure.axis is not None else ("x", "y")
    for node_id in node_ids:
        node_y, node_x = divmod(node_id, nx)
        for axis in axes_to_check:
            for rule in active_by_axis[axis]:
                if _rule_writes_node(
                    rule,
                    node_x,
                    node_y,
                    active_by_axis,
                    axes,
                ):
                    related.add(rule.index)
    return replace(failure, rule_indices=tuple(sorted(related)))


def _rule_writes_node(rule, node_x, node_y, active_by_axis, axes):
    if rule.target == rule.rail_coord:
        return False

    if rule.axis == "x":
        if node_x != rule.rail_axis_index:
            return False
        span_index = node_y
        span_axis = "y"
        opposite_axis = "y"
    else:
        if node_y != rule.rail_axis_index:
            return False
        span_index = node_x
        span_axis = "x"
        opposite_axis = "x"

    span_coord = axes[span_axis][span_index]
    if rule.span_min <= span_coord <= rule.span_max:
        return True

    # Outside the baseline span, this rule writes only when an active
    # perpendicular rule couples the structural rail intersection.
    for opposite in active_by_axis[opposite_axis]:
        if opposite.rail_axis_index != span_index:
            continue
        if (
            rule.span_min <= opposite.target <= rule.span_max
            and opposite.span_min <= rule.target <= opposite.span_max
        ):
            return True
    return False


def _affected_cells(updates, nx, ny):
    cells = set()
    changed_nodes = set(updates["x"])
    changed_nodes.update(updates["y"])
    for node_id in changed_nodes:
        node_y, node_x = divmod(node_id, nx)
        for cell_y in (node_y - 1, node_y):
            if not 0 <= cell_y < ny - 1:
                continue
            for cell_x in (node_x - 1, node_x):
                if 0 <= cell_x < nx - 1:
                    cells.add(cell_y * (nx - 1) + cell_x)
    return cells


def _node_coordinate(node_id, axes, updates, nx):
    node_y, node_x = divmod(node_id, nx)
    return (
        updates["x"].get(node_id, axes["x"][node_x]),
        updates["y"].get(node_id, axes["y"][node_y]),
    )


def _convex_cross_products(vertices):
    crosses = []
    for index in range(4):
        first = vertices[index]
        second = vertices[(index + 1) % 4]
        third = vertices[(index + 2) % 4]
        first_dx = second[0] - first[0]
        first_dy = second[1] - first[1]
        second_dx = third[0] - second[0]
        second_dy = third[1] - second[1]
        crosses.append(
            math.fsum(
                (
                    first_dx * second_dy,
                    -(first_dy * second_dx),
                )
            )
        )
    return tuple(crosses)


def _validate_feature_representation(
    axes,
    active,
    active_by_axis,
    span_bounds,
    updates,
    nx,
    state,
):
    updated_on_rail = _updated_span_indices_by_rail(updates, nx)
    cross_update_candidates = _cross_update_candidates(updates, nx)
    active_by_target = {
        axis: _rules_by_target(active_by_axis[axis])
        for axis in ("x", "y")
    }
    invalid_noop_targets = set()

    # A no-op line can still participate in a runtime coupled corner.  Check
    # those sparse intersection nodes after all moving assignments are known.
    for x_rule, y_rule in _iter_coupled_pairs(active_by_axis):
        node_id = y_rule.rail_axis_index * nx + x_rule.rail_axis_index
        for rule in (x_rule, y_rule):
            if rule.target != rule.rail_coord:
                continue
            x_coord, y_coord = _node_coordinate(node_id, axes, updates, nx)
            coord = x_coord if rule.axis == "x" else y_coord
            if coord != rule.target:
                invalid_noop_targets.add(rule.index)

    for rule in active:
        lo, hi = span_bounds[rule.index]
        all_on_target = rule.index not in invalid_noop_targets
        if all_on_target and rule.target == rule.rail_coord:
            changed_indices = updated_on_rail[rule.axis].get(
                rule.rail_axis_index,
                (),
            )
            position = bisect_left(changed_indices, lo)
            all_on_target = not (
                position < len(changed_indices)
                and changed_indices[position] < hi
            )

        has_min = _endpoint_is_represented(
            rule,
            rule.span_min,
            lo,
            hi,
            axes,
            updates,
            cross_update_candidates,
            active_by_target,
            nx,
        )
        has_max = _endpoint_is_represented(
            rule,
            rule.span_max,
            lo,
            hi,
            axes,
            updates,
            cross_update_candidates,
            active_by_target,
            nx,
        )
        if all_on_target and has_min and has_max:
            continue
        return SnapPlanFailure(
            reason="feature_not_represented",
            state=state,
            detail=(
                f"rule {rule.index} on {rule.axis}-axis does not retain its "
                "exact target coordinate and both span endpoints"
            ),
            axis=rule.axis,
            rail_id=rule.rail_id,
            rule_indices=(rule.index,),
        )
    return None


def _updated_span_indices_by_rail(updates, nx):
    """Index moving updates by structural rail without rule-sized sets."""
    result = {"x": {}, "y": {}}
    for node_id in updates["x"]:
        node_y, node_x = divmod(node_id, nx)
        result["x"].setdefault(node_x, []).append(node_y)
    for node_id in updates["y"]:
        node_y, node_x = divmod(node_id, nx)
        result["y"].setdefault(node_y, []).append(node_x)
    for by_rail in result.values():
        for indices in by_rail.values():
            indices.sort()
    return result


def _cross_update_candidates(updates, nx):
    """Index span-coordinate updates that may preserve a feature endpoint."""
    result = {"x": {}, "y": {}}
    # A y-coordinate update can provide an endpoint for an x-axis rule on the
    # structural x rail at ``node_x`` (and vice versa).
    for node_id, target in updates["y"].items():
        node_y, node_x = divmod(node_id, nx)
        result["x"].setdefault((node_x, target), []).append(node_y)
    for node_id, target in updates["x"].items():
        node_y, node_x = divmod(node_id, nx)
        result["y"].setdefault((node_y, target), []).append(node_x)
    for candidates in result.values():
        for indices in candidates.values():
            indices.sort()
    return result


def _rules_by_target(rules):
    result = {}
    for rule in rules:
        result.setdefault(rule.target, []).append(rule)
    return result


def _endpoint_is_represented(
    rule,
    endpoint,
    lo,
    hi,
    axes,
    updates,
    cross_update_candidates,
    active_by_target,
    nx,
):
    span_axis = "y" if rule.axis == "x" else "x"
    span_values = axes[span_axis]

    station_index = bisect_left(span_values, endpoint, lo, hi)
    if (
        station_index < hi
        and span_values[station_index] == endpoint
        and _rule_node_matches(
            rule,
            station_index,
            endpoint,
            axes,
            updates,
            nx,
        )
    ):
        return True

    # An active perpendicular move can turn an interior structural station
    # into the exact endpoint even when the endpoint was not a baseline axis
    # station.  Query only candidates with this exact target value.
    candidate_indices = cross_update_candidates[rule.axis].get(
        (rule.rail_axis_index, endpoint),
        (),
    )
    position = bisect_left(candidate_indices, lo)
    while position < len(candidate_indices):
        span_index = candidate_indices[position]
        if span_index >= hi:
            break
        if _rule_node_matches(
            rule,
            span_index,
            endpoint,
            axes,
            updates,
            nx,
        ):
            return True
        position += 1

    # Coupled-corner nodes may sit outside the rule's baseline rail span.
    opposite_axis = "y" if rule.axis == "x" else "x"
    for opposite in active_by_target[opposite_axis].get(endpoint, ()):
        if not opposite.span_min <= rule.target <= opposite.span_max:
            continue
        node_id = (
            opposite.rail_axis_index * nx + rule.rail_axis_index
            if rule.axis == "x"
            else rule.rail_axis_index * nx + opposite.rail_axis_index
        )
        x_coord, y_coord = _node_coordinate(node_id, axes, updates, nx)
        coord = x_coord if rule.axis == "x" else y_coord
        span_coord = y_coord if rule.axis == "x" else x_coord
        if coord == rule.target and span_coord == endpoint:
            return True
    return False


def _rule_node_matches(rule, span_index, endpoint, axes, updates, nx):
    if rule.axis == "x":
        node_id = span_index * nx + rule.rail_axis_index
    else:
        node_id = rule.rail_axis_index * nx + span_index
    x_coord, y_coord = _node_coordinate(node_id, axes, updates, nx)
    coord = x_coord if rule.axis == "x" else y_coord
    span_coord = y_coord if rule.axis == "x" else x_coord
    return coord == rule.target and span_coord == endpoint


__all__ = [
    "SnapPlanFailure",
    "SnapPlanValidationResult",
    "SnapPlanZState",
    "UnsafeSnapPlanError",
    "preflight_snap_plan",
    "validate_snap_plan",
]
