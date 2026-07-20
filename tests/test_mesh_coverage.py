import numpy as np
import pytest

from optimal_checkerboard.algorithms.mesh_coverage import validate_mesh_coverage


def _structured_mesh(x_values=(0.0, 1.0, 2.0), y_values=(0.0, 1.0, 2.0)):
    nodes = np.asarray(
        [(x, y) for y in y_values for x in x_values],
        dtype=np.float64,
    )
    nx = len(x_values)
    elements = []
    for row in range(len(y_values) - 1):
        for column in range(nx - 1):
            lower_left = row * nx + column
            elements.append(
                [
                    lower_left,
                    lower_left + 1,
                    lower_left + nx + 1,
                    lower_left + nx,
                ]
            )
    return nodes, np.asarray(elements, dtype=np.intp)


def _rail(axis, coord, rail_id=0):
    return {"axis": axis, "coord": coord, "rail_id": rail_id}


def _rule(axis, rail_coord, span_min, span_max, rail_id=0):
    return {
        "kind": "snap",
        "axis": axis,
        "rail_id": rail_id,
        "rail_coord": rail_coord,
        "span_min": span_min,
        "span_max": span_max,
    }


def test_feature_requires_connected_collinear_mesh_edge_chain():
    nodes, elements = _structured_mesh()
    report = validate_mesh_coverage(
        nodes,
        elements,
        rails={"x": [_rail("x", 1.0)], "y": []},
        snap_rules={0.0: [_rule("x", 1.0, 0.0, 2.0)]},
        chunk_size=1,
    )

    assert report["rule_count"] == 1
    assert report["retained_edge_count"] >= 2


def test_feature_endpoints_without_collinear_edges_are_rejected():
    nodes = np.asarray(
        [
            [1.0, 0.0],
            [2.0, 1.0],
            [1.0, 2.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    elements = np.asarray([[0, 1, 2, 3]], dtype=np.intp)

    with pytest.raises(ValueError, match="no collinear edge chain"):
        validate_mesh_coverage(
            nodes,
            elements,
            rails={"x": [_rail("x", 1.0)], "y": []},
            snap_rules=[_rule("x", 1.0, 0.0, 2.0)],
        )


def test_geometrically_touching_duplicate_nodes_do_not_form_a_chain():
    # The two vertical edges meet geometrically at (1, 1), but use different
    # node ids.  Snapping them as one feature would leave a topological crack.
    nodes = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [2.0, 2.0],
        ],
        dtype=np.float64,
    )
    elements = np.asarray(
        [
            [2, 0, 1, 3],
            [4, 6, 7, 5],
        ],
        dtype=np.intp,
    )

    with pytest.raises(ValueError, match="duplicate or reversed structural nodes"):
        validate_mesh_coverage(
            nodes,
            elements,
            rails={"x": [_rail("x", 1.0)], "y": []},
            snap_rules=[_rule("x", 1.0, 0.0, 2.0)],
        )


def test_rule_span_requires_nodes_at_both_exact_feature_endpoints():
    nodes, elements = _structured_mesh(
        x_values=(0.0, 1.0, 2.0),
        y_values=(0.0, 2.0),
    )

    with pytest.raises(ValueError, match="missing a feature endpoint"):
        validate_mesh_coverage(
            nodes,
            elements,
            rails={"x": [_rail("x", 1.0)], "y": []},
            snap_rules=[_rule("x", 1.0, 0.5, 2.0)],
        )


def test_mandatory_station_cannot_cut_through_an_active_element():
    nodes, elements = _structured_mesh(
        x_values=(0.0, 2.0),
        y_values=(0.0, 1.0),
    )

    with pytest.raises(ValueError, match="crosses the interior"):
        validate_mesh_coverage(
            nodes,
            elements,
            rails={"x": [], "y": []},
            snap_rules=[],
            mandatory_coordinates={"x": [1.0], "y": []},
        )


def test_mandatory_station_is_complete_on_an_element_boundary():
    nodes, elements = _structured_mesh(
        x_values=(0.0, 1.0, 2.0),
        y_values=(0.0, 1.0),
    )

    report = validate_mesh_coverage(
        nodes,
        elements,
        rails={"x": [], "y": []},
        snap_rules=[],
        mandatory_coordinates={"x": [1.0], "y": []},
    )

    assert report["mandatory_station_count"] == 1


def test_mandatory_station_must_be_part_of_active_topology():
    nodes, elements = _structured_mesh(
        x_values=(0.0, 1.0),
        y_values=(0.0, 1.0),
    )

    with pytest.raises(ValueError, match="no positive-length active mesh edge"):
        validate_mesh_coverage(
            nodes,
            elements,
            rails={"x": [], "y": []},
            snap_rules=[],
            mandatory_coordinates={"x": [2.0], "y": []},
        )


def test_rule_rail_metadata_must_be_consistent():
    nodes, elements = _structured_mesh()

    with pytest.raises(ValueError, match="disagrees with rail metadata"):
        validate_mesh_coverage(
            nodes,
            elements,
            rails={"x": [_rail("x", 1.0)], "y": []},
            snap_rules=[_rule("x", 1.5, 0.0, 2.0)],
        )
