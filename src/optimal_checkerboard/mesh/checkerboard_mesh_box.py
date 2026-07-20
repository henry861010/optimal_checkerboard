"""Generate structured checkerboard meshes for rectangular domains."""

from numbers import Real

import numpy as np


_CONNECTIVITY_CHUNK_SIZE = 250_000
_INT32_MAX = np.iinfo(np.int32).max


def checkerboard_mesh_box(x_list, y_list, element_size, return_axes=False):
    """Generate a structured quadrilateral checkerboard mesh.

    Args:
        x_list: Sorted x coordinates that must appear as mesh grid lines.
        y_list: Sorted y coordinates that must appear as mesh grid lines.
        element_size: Maximum preferred element size inside each interval.

    Returns:
        A tuple containing node coordinates and quadrilateral connectivity.
        When ``return_axes`` is true, the one-dimensional x/y node arrays are
        appended to the tuple.  These arrays let callers index structured rail
        nodes without scanning the full node and element arrays.
    """
    x_nodes, y_nodes = checkerboard_mesh_box_axes(
        x_list,
        y_list,
        element_size,
    )

    nodes = _mesh_nodes(x_nodes, y_nodes)
    elements = _mesh_elements(len(x_nodes), len(y_nodes))
    if return_axes:
        return nodes, elements, x_nodes, y_nodes
    return nodes, elements


def checkerboard_mesh_box_axes(x_list, y_list, element_size):
    """Return only structured axis nodes for bounded safety preflight."""
    element_size = _positive_finite_number(element_size, "element_size")
    x_list = _validated_axis_list(x_list, "x_list")
    y_list = _validated_axis_list(y_list, "y_list")
    nx_elems = _interval_element_counts(x_list, element_size)
    ny_elems = _interval_element_counts(y_list, element_size)
    nx = sum(int(count) for count in nx_elems) + 1
    ny = sum(int(count) for count in ny_elems) + 1
    _validate_int32_connectivity_capacity(nx, ny)
    return _axis_nodes(x_list, nx_elems), _axis_nodes(y_list, ny_elems)


def _interval_element_counts(axis_list, element_size):
    """Return representable subdivision counts without integer overflow."""
    counts = np.maximum(1.0, np.ceil(np.diff(axis_list) / element_size))
    if np.any(~np.isfinite(counts)) or np.any(counts > _INT32_MAX):
        raise OverflowError(
            "structured mesh exceeds int32 connectivity capacity"
        )
    return counts.astype(np.int64)


def _validated_axis_list(values, name):
    """Return one finite, strictly increasing mesh-axis coordinate array."""
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
        raise ValueError(f"{name} intervals must be representable and finite")
    if np.any(intervals <= 0):
        raise ValueError(f"{name} coordinates must be strictly increasing")
    return values


def _positive_finite_number(value, name):
    """Return a positive finite float used by structured mesh generation."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


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
    nx = len(x_nodes)
    ny = len(y_nodes)
    node_count = _validate_int32_connectivity_capacity(nx, ny)
    nodes = np.zeros((node_count, 3), dtype=np.float64)

    # Assign through a view of the final array.  NumPy broadcasts the two axis
    # arrays directly into their columns, avoiding two full-size meshgrid
    # temporaries for large checkerboards.
    node_grid = nodes.reshape(ny, nx, 3)
    node_grid[:, :, 0] = np.asarray(x_nodes)[None, :]
    node_grid[:, :, 1] = np.asarray(y_nodes)[:, None]
    return nodes


def _mesh_elements(nx, ny, chunk_size=_CONNECTIVITY_CHUNK_SIZE):
    """Create quadrilateral connectivity for a structured node grid."""
    if not isinstance(chunk_size, (int, np.integer)) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    _validate_int32_connectivity_capacity(nx, ny)
    element_count = (ny - 1) * (nx - 1)
    elements = np.empty((element_count, 4), dtype=np.int32)
    elements_per_row = nx - 1
    for start in range(0, element_count, int(chunk_size)):
        stop = min(start + int(chunk_size), element_count)
        # A flattened element id differs from its lower-left node id by one
        # skipped node at the end of each completed element row.  Materialize
        # only this bounded vector, then write connectivity into final storage.
        lower_left = np.arange(start, stop, dtype=np.int32)
        lower_left += lower_left // elements_per_row
        destination = elements[start:stop]
        destination[:, 0] = lower_left
        destination[:, 1] = lower_left + 1
        destination[:, 2] = lower_left + nx + 1
        destination[:, 3] = lower_left + nx
    return elements


def _validate_int32_connectivity_capacity(nx, ny):
    """Return node count or fail before allocating an unindexable grid."""
    nx = int(nx)
    ny = int(ny)
    if nx < 2 or ny < 2:
        raise ValueError("structured mesh axes must each contain two nodes")
    node_count = nx * ny
    if node_count - 1 > _INT32_MAX:
        raise OverflowError(
            "structured mesh exceeds int32 connectivity capacity"
        )
    return node_count
