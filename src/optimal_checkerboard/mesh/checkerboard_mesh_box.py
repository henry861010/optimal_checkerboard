"""Generate structured checkerboard meshes for rectangular domains."""

import numpy as np


def checkerboard_mesh_box(x_list, y_list, element_size):
    """Generate a structured quadrilateral checkerboard mesh.

    Args:
        x_list: Sorted x coordinates that must appear as mesh grid lines.
        y_list: Sorted y coordinates that must appear as mesh grid lines.
        element_size: Maximum preferred element size inside each interval.

    Returns:
        A tuple containing node coordinates and quadrilateral connectivity.
    """
    x_list = np.asarray(x_list)
    y_list = np.asarray(y_list)

    nx_elems = np.maximum(
        1,
        np.ceil(np.diff(x_list) / element_size).astype(int),
    )
    ny_elems = np.maximum(
        1,
        np.ceil(np.diff(y_list) / element_size).astype(int),
    )

    x_nodes = _axis_nodes(x_list, nx_elems)
    y_nodes = _axis_nodes(y_list, ny_elems)

    nodes = _mesh_nodes(x_nodes, y_nodes)
    elements = _mesh_elements(len(x_nodes), len(y_nodes))
    return nodes, elements


def _axis_nodes(axis_list, interval_element_counts):
    """Generate all node coordinates along one structured mesh axis."""
    node_count = interval_element_counts.sum() + 1
    axis_nodes = np.zeros(node_count)

    current_index = 0
    axis_nodes[0] = axis_list[0]
    for index in range(len(axis_list) - 1):
        element_count = interval_element_counts[index]
        interval_nodes = np.linspace(
            axis_list[index],
            axis_list[index + 1],
            element_count + 1,
        )
        start = current_index + 1
        stop = start + element_count
        axis_nodes[start:stop] = interval_nodes[1:]
        current_index += element_count

    return axis_nodes


def _mesh_nodes(x_nodes, y_nodes):
    """Create flattened 3D node coordinates from x and y grid coordinates."""
    node_count = len(y_nodes) * len(x_nodes)
    nodes = np.zeros((node_count, 3), dtype=np.float64)

    x_grid, y_grid = np.meshgrid(x_nodes, y_nodes)
    nodes[:, 0] = x_grid.ravel()
    nodes[:, 1] = y_grid.ravel()
    return nodes


def _mesh_elements(nx, ny):
    """Create quadrilateral connectivity for a structured node grid."""
    element_count = (ny - 1) * (nx - 1)
    node_ids = np.arange(ny * nx, dtype=np.int32).reshape(ny, nx)

    elements = np.empty((element_count, 4), dtype=np.int32)
    elements[:, 0] = node_ids[:-1, :-1].ravel()
    elements[:, 1] = node_ids[:-1, 1:].ravel()
    elements[:, 2] = node_ids[1:, 1:].ravel()
    elements[:, 3] = node_ids[1:, :-1].ravel()
    return elements
