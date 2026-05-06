"""Client example for the 2.5D shared-rail checkerboard mesher."""

import math
import os
import sys
import tempfile


def _add_src_to_path():
    """Add the local src directory to sys.path for direct script execution."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src_root = os.path.join(repo_root, "src")
    sys.path.insert(0, src_root)


def _example_faces():
    face1 = {
        "type": "BOX",
        "dim": [0, 0, 0, 10, 10, 0],
    }
    face2 = {
        "type": "BOX",
        "dim": [11, 12, 0, 18, 18, 0],
    }
    face3 = {
        "type": "BOX",
        "dim": [9, 30, 0, 20, 40, 10],
    }
    return [face1, face2, face3]


def _collect_points(value, points):
    """Append every xyz point found in a nested face dimension object."""
    if (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(isinstance(item, (int, float)) for item in value)
    ):
        points.append(value)
        return

    if isinstance(value, (list, tuple)):
        for item in value:
            _collect_points(item, points)


def _face_bounds(faces, offset=1):
    """Return xy bounds that cover all example faces."""
    points = []
    for face in faces:
        if face["type"] == "BOX":
            x1, y1, _, x2, y2, _ = face["dim"]
            points.extend([(x1, y1, 0), (x2, y2, 0)])
        elif face["type"] == "LINE":
            points.extend(face["dim"])
        elif face["type"] == "POLYGON":
            _collect_points(face["dim"], points)
        else:
            raise ValueError(f"Unsupported face type: {face['type']}")

    if not points:
        raise ValueError("At least one face point is required")

    x_values = [float(point[0]) for point in points]
    y_values = [float(point[1]) for point in points]
    return [min(x_values) - offset, min(y_values) - offset, max(x_values) + offset, max(y_values) + offset]


def _format_z_value(z_value):
    """Return compact labels for z values used in plot titles."""
    return f"{float(z_value):g}"


def _edge_key(point_a, point_b, decimals=8):
    """Return an orientation-independent key for one plotted mesh edge."""
    start = tuple(round(float(value), decimals) for value in point_a[:2])
    end = tuple(round(float(value), decimals) for value in point_b[:2])
    return tuple(sorted((start, end)))


def _mesh_edge_segments(nodes, elements):
    """Return unique line segments from quadrilateral node connectivity."""
    segments = []
    seen_edges = set()
    for element in elements:
        points = nodes[element, :2]
        for point_a, point_b in (
            (points[0], points[1]),
            (points[1], points[2]),
            (points[2], points[3]),
            (points[3], points[0]),
        ):
            edge_key = _edge_key(point_a, point_b)
            if edge_key[0] == edge_key[1] or edge_key in seen_edges:
                continue

            seen_edges.add(edge_key)
            segments.append([point_a, point_b])
    return segments


def _plot_mesh_layers(layers, output_path):
    """Save one figure that visualizes every snapped mesh layer."""
    try:
        mpl_cache_dir = os.path.join(
            tempfile.gettempdir(),
            "optimal_checkerboard_matplotlib",
        )
        os.makedirs(mpl_cache_dir, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", mpl_cache_dir)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
    except ModuleNotFoundError:
        print(
            "matplotlib is required to plot mesh layers. "
            "Install it with: python -m pip install matplotlib"
        )
        print("Skip mesh layer plot.")
        return False

    if not layers:
        print("No z layers found; skip mesh layer plot.")
        return False

    column_count = min(3, len(layers))
    row_count = math.ceil(len(layers) / column_count)
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5 * column_count, 5 * row_count),
        squeeze=False,
    )
    axes = axes.ravel()

    for axis, layer in zip(axes, layers):
        nodes = layer["nodes"]
        elements = layer["elements"]
        segments = _mesh_edge_segments(nodes, elements)
        points = nodes[:, :2]
        axis.add_collection(
            LineCollection(segments, colors="#2f4f6f", linewidths=0.9)
        )
        axis.scatter(
            points[:, 0],
            points[:, 1],
            s=12,
            color="#d45d45",
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )
        axis.set_title(
            f"z={_format_z_value(layer['z'])}"
        )
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        axis.set_aspect("equal", adjustable="box")
        axis.autoscale()
        axis.margins(0.08)
        axis.grid(True, color="#d9d9d9", linewidth=0.5)

    for axis in axes[len(layers):]:
        axis.axis("off")

    fig.suptitle("Snapped checkerboard mesh layers", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def main():
    """Run a complete mesher setup and snap-rule application example."""
    _add_src_to_path()

    from optimal_checkerboard import OptimalMesh25D

    mesher = OptimalMesh25D()
    faces = _example_faces()
    group_lines_v, group_lines_h, x_list, y_list = mesher.set_pattern(
        faces,
        element_size=11.0,
        ratio=0.2,
    )
    mesh2d = mesher.mesh_checkerboard_box(_face_bounds(faces))

    print()
    print("Shared checkerboard x rails:", x_list)
    print("Shared checkerboard y rails:", y_list)
    print("Vertical rail groups:", len(group_lines_v))
    print("Horizontal rail groups:", len(group_lines_h))
    #print("2D nodes:", mesh2d.nodes.shape)
    #print("2D elements:", mesh2d.elements.shape)
    print()

    layers = []
    layers.append(
        {
            "z": -1,
            "nodes": mesh2d.nodes.copy(),
            "elements": mesh2d.elements.copy(),
        }
    )
    for z_value in sorted(mesher.get_snap_rules()):
        mesher.apply_snap_rules_at_z(z_value)
        layers.append(
            {
                "z": z_value,
                "nodes": mesh2d.nodes.copy(),
                "elements": mesh2d.elements.copy(),
            }
        )

    output_path = os.path.join(os.path.dirname(__file__), "mesh_layers.png")
    if _plot_mesh_layers(layers, output_path):
        print("Mesh layer plot:", output_path)

if __name__ == "__main__":
    main()
