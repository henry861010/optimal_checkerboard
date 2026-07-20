"""Dispatch checkerboard mesh generation by footprint domain type."""

import numpy as np

from optimal_checkerboard.mesh.checkerboard_mesh_box import (
    checkerboard_mesh_box,
    checkerboard_mesh_box_axes,
)
from optimal_checkerboard.mesh.checkerboard_mesh_cylinder import (
    checkerboard_mesh_cylinder,
)
from optimal_checkerboard.mesh.domain import mesh_domain_pinned_coordinates


def generate_checkerboard_mesh(
    domain,
    x_list,
    y_list,
    element_size,
    mandatory_coordinates=None,
    return_metadata=False,
):
    """Generate a 2D checkerboard mesh for a normalized footprint domain."""
    domain_type = domain["type"]
    if domain_type == "BOX":
        return generate_checkerboard_mesh_box(
            domain,
            x_list,
            y_list,
            element_size,
            mandatory_coordinates=mandatory_coordinates,
            return_metadata=return_metadata,
        )
    if domain_type == "CYLINDER":
        return checkerboard_mesh_cylinder(domain, x_list, y_list, element_size)
    raise NotImplementedError(
        f"{domain_type} checkerboard mesh generation is not implemented yet"
    )


def generate_checkerboard_mesh_box(
    domain,
    x_list,
    y_list,
    element_size,
    mandatory_coordinates=None,
    return_metadata=False,
):
    """Generate a structured checkerboard mesh for a BOX footprint domain."""
    if domain["type"] != "BOX":
        raise ValueError("BOX checkerboard mesh generation requires a BOX domain")

    mandatory_coordinates = _normalize_axis_coordinates(
        mandatory_coordinates,
        "mandatory_coordinates",
    )
    mesh_x_list, mesh_y_list = mesh_lists_for_box_domain(
        domain,
        x_list,
        y_list,
        mandatory_coordinates=mandatory_coordinates,
    )
    mesh_result = checkerboard_mesh_box(
        mesh_x_list,
        mesh_y_list,
        element_size,
        return_axes=return_metadata,
    )

    if return_metadata:
        nodes, elements, x_nodes, y_nodes = mesh_result
        metadata = _structured_mesh_metadata(
            domain,
            x_nodes,
            y_nodes,
            mandatory_coordinates,
        )
        return nodes, elements, mesh_x_list, mesh_y_list, metadata

    nodes, elements = mesh_result
    return nodes, elements, mesh_x_list, mesh_y_list


def structured_axes_for_box_domain(
    domain,
    x_list,
    y_list,
    element_size,
    mandatory_coordinates=None,
):
    """Return BOX required stations and final 1D nodes without a 2D grid."""
    if domain["type"] != "BOX":
        raise ValueError("structured BOX axes require a BOX domain")
    mandatory_coordinates = _normalize_axis_coordinates(
        mandatory_coordinates,
        "mandatory_coordinates",
    )
    mesh_x_list, mesh_y_list = mesh_lists_for_box_domain(
        domain,
        x_list,
        y_list,
        mandatory_coordinates=mandatory_coordinates,
    )
    x_nodes, y_nodes = checkerboard_mesh_box_axes(
        mesh_x_list,
        mesh_y_list,
        element_size,
    )
    return x_nodes, y_nodes, mesh_x_list, mesh_y_list


def mesh_lists_for_box_domain(
    domain,
    x_list,
    y_list,
    mandatory_coordinates=None,
    eps=0.0,
):
    """Return rail coordinate lists bounded by a BOX footprint."""
    eps = _non_negative_finite_number(eps, "eps")
    try:
        xmin, ymin, xmax, ymax = domain["bbox"]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("BOX domain must provide four bbox coordinates") from exc
    xmin, ymin, xmax, ymax = [
        _finite_number(value, "BOX domain boundary")
        for value in (xmin, ymin, xmax, ymax)
    ]
    if xmin >= xmax or ymin >= ymax:
        raise ValueError("BOX domain must have positive area")
    rail_coordinates = _normalize_axis_coordinates(
        {"x": x_list, "y": y_list},
        "rail coordinates",
    )
    mandatory_coordinates = _normalize_axis_coordinates(
        mandatory_coordinates,
        "mandatory_coordinates",
    )
    x_values = rail_coordinates["x"] + mandatory_coordinates["x"]
    y_values = rail_coordinates["y"] + mandatory_coordinates["y"]

    _validate_rails_inside_bounds(x_values, xmin, xmax, "x", eps=eps)
    _validate_rails_inside_bounds(y_values, ymin, ymax, "y", eps=eps)

    x_values += [xmin, xmax]
    y_values += [ymin, ymax]
    return _unique_sorted(x_values, eps=eps), _unique_sorted(y_values, eps=eps)


def build_structured_rail_node_index(
    x_nodes,
    y_nodes,
    rails_by_axis,
    eps=0.0,
):
    """Build rail-node lookups directly from structured mesh axis arrays.

    The returned dictionaries match ``OptimalMesh25D.rail_node_index`` while
    avoiding the full-element ``unique`` pass and global coordinate sorts used
    for arbitrary assigned meshes.
    """
    eps = _non_negative_finite_number(eps, "eps")
    x_nodes = _validated_structured_axis(x_nodes, "x_nodes")
    y_nodes = _validated_structured_axis(y_nodes, "y_nodes")
    if not isinstance(rails_by_axis, dict):
        raise ValueError("rails_by_axis must be a dictionary")

    return {
        "x": _structured_axis_rail_indices(
            x_nodes,
            y_nodes,
            rails_by_axis.get("x", ()),
            axis="x",
            eps=eps,
        ),
        "y": _structured_axis_rail_indices(
            x_nodes,
            y_nodes,
            rails_by_axis.get("y", ()),
            axis="y",
            eps=eps,
        ),
    }


def _structured_mesh_metadata(
    domain,
    x_nodes,
    y_nodes,
    mandatory_coordinates,
):
    """Return compact topology metadata for one structured BOX mesh."""
    domain_pinned = mesh_domain_pinned_coordinates(domain)
    pinned = {
        axis: _unique_sorted(
            domain_pinned[axis] + mandatory_coordinates[axis],
            eps=0.0,
        )
        for axis in ("x", "y")
    }
    return {
        "kind": "STRUCTURED_BOX",
        "grid_shape": (len(y_nodes), len(x_nodes)),
        "x_nodes": x_nodes,
        "y_nodes": y_nodes,
        "mandatory_axis_coordinates": {
            axis: np.asarray(mandatory_coordinates[axis], dtype=np.float64)
            for axis in ("x", "y")
        },
        "pinned_axis_coordinates": pinned,
    }


def _structured_axis_rail_indices(x_nodes, y_nodes, rails, axis, eps):
    """Return structured node-id arithmetic for one rail axis."""
    coord_nodes = x_nodes if axis == "x" else y_nodes
    span_nodes = y_nodes if axis == "x" else x_nodes
    nx = len(x_nodes)
    ny = len(y_nodes)
    rail_indices = []

    for rail in rails:
        coord = rail.get("coord") if isinstance(rail, dict) else rail
        coord = _finite_number(coord, f"{axis}-axis rail coordinate")
        coord_index = _matching_axis_index(coord_nodes, coord, eps=eps)
        if coord_index is None:
            node_ids = np.empty(0, dtype=np.intp)
            span_values = np.empty(0, dtype=np.float64)
        elif axis == "x":
            node_ids = coord_index + np.arange(ny, dtype=np.intp) * nx
            span_values = span_nodes
        else:
            node_ids = coord_index * nx + np.arange(nx, dtype=np.intp)
            span_values = span_nodes

        rail_indices.append(
            {
                "coord": coord,
                "node_ids": node_ids,
                "span_values": span_values,
            }
        )
    return rail_indices


def _matching_axis_index(axis_nodes, coord, eps):
    """Return the nearest structured coordinate index within tolerance."""
    insertion = int(np.searchsorted(axis_nodes, coord, side="left"))
    candidates = []
    if insertion < len(axis_nodes):
        candidates.append(insertion)
    if insertion:
        candidates.append(insertion - 1)
    if not candidates:
        return None

    best_index = min(
        candidates,
        key=lambda index: (abs(float(axis_nodes[index]) - coord), index),
    )
    if abs(float(axis_nodes[best_index]) - coord) > eps:
        return None
    return best_index


def _validate_rails_inside_bounds(values, min_value, max_value, axis, eps):
    outside_values = [
        float(value)
        for value in values
        if float(value) < min_value - eps or float(value) > max_value + eps
    ]
    if outside_values:
        raise ValueError(
            f"{axis}-axis checkerboard rails fall outside the root footprint"
        )


def _unique_sorted(values, eps=0.0):
    """Return sorted floats with only exactly equal values removed.

    ``eps`` remains in the private signature for compatibility with existing
    callers, but it must not collapse distinct geometry coordinates.
    """
    values = sorted(
        _finite_number(value, "mesh-axis coordinate")
        for value in values
    )
    if not values:
        return np.asarray([], dtype=np.float64)

    unique_values = [values[0]]
    for value in values[1:]:
        if value != unique_values[-1]:
            unique_values.append(value)
    return np.asarray(unique_values, dtype=np.float64)


def _normalize_axis_coordinates(values, name):
    """Return finite x/y coordinate lists from an optional mapping."""
    if values is None:
        return {"x": [], "y": []}
    if not isinstance(values, dict):
        raise ValueError(f"{name} must be a dictionary with x/y lists")

    unknown_axes = set(values) - {"x", "y"}
    if unknown_axes:
        raise ValueError(f"{name} contains unsupported axes")
    return {
        axis: sorted(
            {
                _finite_number(value, f"{axis}-axis coordinate")
                for value in values.get(axis, ())
            }
        )
        for axis in ("x", "y")
    }


def _validated_structured_axis(values, name):
    """Return a finite, strictly increasing structured axis array."""
    try:
        values = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric coordinates") from exc
    if values.ndim != 1 or len(values) < 2:
        raise ValueError(f"{name} must contain at least two coordinates")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} coordinates must be finite")
    intervals = np.diff(values)
    if np.any(~np.isfinite(intervals)):
        raise ValueError(f"{name} intervals must be finite")
    if np.any(intervals <= 0):
        raise ValueError(f"{name} coordinates must be strictly increasing")
    return values


def _non_negative_finite_number(value, name):
    value = _finite_number(value, name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _finite_number(value, name):
    """Return one finite floating-point input coordinate."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value
