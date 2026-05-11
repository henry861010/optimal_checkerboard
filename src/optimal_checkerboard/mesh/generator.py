"""Dispatch checkerboard mesh generation by footprint domain type."""

import numpy as np

from optimal_checkerboard.mesh.checkerboard_mesh_box import checkerboard_mesh_box
from optimal_checkerboard.mesh.checkerboard_mesh_cylinder import (
    checkerboard_mesh_cylinder,
)


def generate_checkerboard_mesh(domain, x_list, y_list, element_size):
    """Generate a 2D checkerboard mesh for a normalized footprint domain."""
    domain_type = domain["type"]
    if domain_type == "BOX":
        return generate_checkerboard_mesh_box(
            domain,
            x_list,
            y_list,
            element_size,
        )
    if domain_type == "CYLINDER":
        return checkerboard_mesh_cylinder(domain, x_list, y_list, element_size)
    raise NotImplementedError(
        f"{domain_type} checkerboard mesh generation is not implemented yet"
    )


def generate_checkerboard_mesh_box(domain, x_list, y_list, element_size):
    """Generate a structured checkerboard mesh for a BOX footprint domain."""
    if domain["type"] != "BOX":
        raise ValueError("BOX checkerboard mesh generation requires a BOX domain")

    mesh_x_list, mesh_y_list = mesh_lists_for_box_domain(domain, x_list, y_list)
    nodes, elements = checkerboard_mesh_box(
        mesh_x_list,
        mesh_y_list,
        element_size,
    )
    return nodes, elements, mesh_x_list, mesh_y_list


def mesh_lists_for_box_domain(domain, x_list, y_list, eps=1e-9):
    """Return rail coordinate lists bounded by a BOX footprint."""
    xmin, ymin, xmax, ymax = domain["bbox"]
    _validate_rails_inside_bounds(x_list, xmin, xmax, "x", eps=eps)
    _validate_rails_inside_bounds(y_list, ymin, ymax, "y", eps=eps)

    x_values = list(x_list) + [xmin, xmax]
    y_values = list(y_list) + [ymin, ymax]
    return _unique_sorted(x_values, eps=eps), _unique_sorted(y_values, eps=eps)


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


def _unique_sorted(values, eps=1e-9):
    """Return sorted float values with near-duplicates removed."""
    values = sorted(float(value) for value in values)
    if not values:
        return np.asarray([], dtype=np.float64)

    unique_values = [values[0]]
    for value in values[1:]:
        if abs(value - unique_values[-1]) > eps:
            unique_values.append(value)
    return np.asarray(unique_values, dtype=np.float64)
