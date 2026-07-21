import os
import sys
import unittest
from unittest import mock

SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
sys.path.insert(0, SRC_ROOT)

import numpy as np

from optimal_checkerboard import OptimalMesh25D
from optimal_checkerboard.algorithms.drag import Dragger
from optimal_checkerboard.data_structure.geometry import Obj


def _area():
    return {
        "type": "BOX",
        "dim": [0, 0, 3, 3],
        "material": "CORE",
    }


def _build_zero_copy_mesher():
    faces = [
        {
            "type": "LINE",
            "dim": [1, 0, 1, 3],
            "bottom_z": 0,
            "top_z": 10,
        },
        {
            "type": "LINE",
            "dim": [1.5, 0, 1.5, 3],
            "bottom_z": 20,
            "top_z": 30,
        },
    ]

    mesher = OptimalMesh25D()
    mesher._set_pattern(
        faces,
        element_size=5,
        ratio=0.2,
        mesh_domain={"type": "BOX", "dim": [0, 0, 3, 3]},
    )
    mesher.mesh_checkerboard()
    return mesher


class TestOptimalMeshBuild(unittest.TestCase):
    @staticmethod
    def _integrity_mesher():
        mesher = OptimalMesh25D()
        mesher._set_pattern(
            [{
                "type": "BOX",
                "dim": [0.0, 0.0, 3.0, 3.0],
                "bottom_z": 0.0,
                "top_z": 1.0,
            }],
            element_size=1.0,
            ratio=0.0,
            mesh_domain={"type": "BOX", "dim": [0.0, 0.0, 3.0, 3.0]},
        )
        mesher.mesh_checkerboard()
        return mesher

    @staticmethod
    def _integrity_stack():
        return [[
            {
                "z": 0.0,
                "areas": [{
                    "type": "BOX",
                    "dim": [0.0, 0.0, 3.0, 3.0],
                    "material": "CORE",
                }],
                "element_size": 1.0,
            },
            {"z": 1.0},
        ]]

    def test_build_rejects_replaced_connectivity_after_indexing(self):
        mesher = self._integrity_mesher()
        mesher.mesh2d.elements = np.delete(
            mesher.mesh2d.elements,
            4,
            axis=0,
        )

        with self.assertRaisesRegex(RuntimeError, "arrays were replaced"):
            mesher.build(self._integrity_stack())

    def test_replaced_arrays_are_not_normalized_before_build_rejects(self):
        mesher = self._integrity_mesher()
        replacement_nodes = mesher.mesh2d.nodes.astype(np.float32)
        replacement_elements = mesher.mesh2d.elements.astype(np.int64)
        mesher.mesh2d.nodes = replacement_nodes
        mesher.mesh2d.elements = replacement_elements

        with self.assertRaisesRegex(RuntimeError, "arrays were replaced"):
            mesher.build(self._integrity_stack())

        self.assertIs(mesher.mesh2d.nodes, replacement_nodes)
        self.assertIs(mesher.mesh2d.elements, replacement_elements)
        self.assertEqual(mesher.mesh2d.nodes.dtype, np.float32)
        self.assertEqual(mesher.mesh2d.elements.dtype, np.int64)

    def test_build_rejects_in_place_connectivity_mutation(self):
        mesher = self._integrity_mesher()
        self.assertFalse(mesher.mesh2d.elements.flags.writeable)
        mesher.mesh2d.elements.setflags(write=True)
        mesher.mesh2d.elements[4] = mesher.mesh2d.elements[0]

        with self.assertRaisesRegex(RuntimeError, "changed after validation"):
            mesher.build(self._integrity_stack())

    def test_build_rejects_unmanaged_in_place_node_mutation(self):
        mesher = self._integrity_mesher()
        center_id = int(
            np.flatnonzero(
                (mesher.mesh2d.nodes[:, 0] == 1.0)
                & (mesher.mesh2d.nodes[:, 1] == 1.0)
            )[0]
        )
        mesher.mesh2d.nodes[center_id, 0] += 0.125

        with self.assertRaisesRegex(RuntimeError, "changed after validation"):
            mesher.build(self._integrity_stack())

    def test_preserved_build_rejects_corrupted_managed_snap_coordinate(self):
        mesher = _build_zero_copy_mesher()
        mesher.apply_snap_rules_at_z(0.0)
        node_id = int(mesher._current_changed_node_ids["x"][0])
        mesher.mesh2d.nodes[node_id, 0] += 0.125
        corrupted = mesher.mesh2d.nodes.copy()

        with self.assertRaisesRegex(RuntimeError, "outside managed snap"):
            mesher.build(
                [[
                    {"z": 0.0, "areas": [_area()], "element_size": 10.0},
                    {"z": 10.0},
                ]],
                preserve_mesh2d=True,
            )

        np.testing.assert_array_equal(mesher.mesh2d.nodes, corrupted)

    def test_preserved_build_does_not_hide_unmanaged_rail_edit(self):
        mesher = _build_zero_copy_mesher()
        rail = mesher.rail_node_index["x"][0]
        node_id = int(rail["node_ids"][0])
        mesher.mesh2d.nodes[node_id, 0] += 0.125
        corrupted = mesher.mesh2d.nodes.copy()

        with self.assertRaisesRegex(RuntimeError, "changed after validation"):
            mesher.build(
                [[
                    {"z": 0.0, "areas": [_area()], "element_size": 10.0},
                    {"z": 10.0},
                ]],
                preserve_mesh2d=True,
            )

        np.testing.assert_array_equal(mesher.mesh2d.nodes, corrupted)

    def test_build_rejects_fail_open_material_schemas(self):
        invalid_metals = [
            {"type": "BAD", "material": "M1"},
            {"type": "normal", "material": "M1", "density": 50.0},
            {"type": "NORMAL", "material": "M1", "density": np.nan},
            {"type": "NORMAL", "material": "M1", "density": np.inf},
            {"type": "NORMAL", "material": "M1", "density": -1.0},
            {"type": "NORMAL", "material": "M1", "density": 100.1},
            {"type": "NORMAL", "material": "M1", "density": "50"},
            {"type": "NORMAL", "material": "M1"},
            {"type": "CONTINUE", "material": ""},
            {"type": "CONVERT", "material": "M2"},
            {
                "type": "CONTINUE",
                "material": "M1",
                "ranges": {"type": "BOX", "dim": [0.0, 0.0, 1.0, 1.0]},
            },
        ]
        for metal in invalid_metals:
            with self.subTest(metal=metal):
                mesher = self._integrity_mesher()
                stack = self._integrity_stack()
                stack[0][0]["areas"][0]["metals"] = [metal]

                with self.assertRaisesRegex(
                    ValueError,
                    "type must|density must|material.*must|ranges must",
                ):
                    mesher.build(stack)

    def test_build_requires_base_area_material_label(self):
        mesher = self._integrity_mesher()
        stack = self._integrity_stack()
        del stack[0][0]["areas"][0]["material"]

        with self.assertRaisesRegex(ValueError, "material must"):
            mesher.build(stack)

    def test_build_rejects_reserved_empty_material_labels(self):
        for field in ("area", "metal", "material_o"):
            with self.subTest(field=field):
                mesher = self._integrity_mesher()
                stack = self._integrity_stack()
                area = stack[0][0]["areas"][0]
                if field == "area":
                    area["material"] = "EMPTY"
                elif field == "metal":
                    area["metals"] = [{
                        "type": "NORMAL",
                        "material": "EMPTY",
                        "density": 50.0,
                    }]
                else:
                    area["metals"] = [{
                        "type": "CONVERT",
                        "material": "M2",
                        "material_o": "EMPTY",
                    }]

                with self.assertRaisesRegex(ValueError, "reserved EMPTY"):
                    mesher.build(stack, preserve_mesh2d=True)

    def test_build_rejects_one_shot_area_holes(self):
        mesher = self._integrity_mesher()
        stack = self._integrity_stack()
        hole = {"type": "BOX", "dim": [1.0, 1.0, 2.0, 2.0]}
        stack[0][0]["areas"][0]["holes"] = (
            item for item in [hole]
        )

        with self.assertRaisesRegex(ValueError, "holes must"):
            mesher.build(stack)

    def test_build_rejects_noncanonical_selector_type_before_mutation(self):
        mesher = self._integrity_mesher()
        stack = self._integrity_stack()
        stack[0][0]["areas"][0]["type"] = "box"
        baseline = mesher.mesh2d.nodes.copy()

        with self.assertRaisesRegex(ValueError, "exactly BOX or POLYGON"):
            mesher.build(stack)

        np.testing.assert_array_equal(mesher.mesh2d.nodes, baseline)

    def test_build_accepts_explicit_none_metals_as_empty(self):
        mesher = self._integrity_mesher()
        stack = self._integrity_stack()
        stack[0][0]["areas"][0]["metals"] = None

        dragger = mesher.build(stack, preserve_mesh2d=True)

        self.assertEqual(dragger.element_num, 9)

    def test_build_options_require_actual_booleans(self):
        for option_name in ("preserve_mesh2d", "allow_independent_bodies"):
            with self.subTest(option_name=option_name):
                mesher = self._integrity_mesher()
                options = {option_name: "False"}

                with self.assertRaisesRegex(ValueError, "must be a boolean"):
                    mesher.build(self._integrity_stack(), **options)

    def test_plain_full_domain_layers_reserve_exact_final_capacity(self):
        mesher = self._integrity_mesher()
        area = {
            "type": "BOX",
            "dim": [0.0, 0.0, 3.0, 3.0],
            "material": "CORE",
        }
        stack = [
            {
                "z": float(z_value),
                "areas": [area],
                "element_size": 1.0,
            }
            for z_value in range(5)
        ] + [{"z": 5.0}]

        dragger = mesher.build([stack], preserve_mesh2d=True)

        self.assertEqual(dragger.element_num, 45)
        self.assertEqual(dragger.node_num, 96)
        self.assertEqual(len(dragger.elements), dragger.element_num)
        self.assertEqual(len(dragger.nodes), dragger.node_num)

    def test_example_children_overlay_main_without_removing_its_top_slab(self):
        """Regression for script/example_geometry.py's material stack."""
        main_dim = [0.0, 0.0, 100.0, 100.0]
        child1_dim = [80.0, 80.0, 90.0, 90.0]
        child2_dim = [70.0, 67.0, 79.9, 77.0]

        main = Obj(type="BOX", dim=main_dim, z=0.0)
        main.add_layer(thk=100.0, material="EMPTY")
        child1 = Obj(type="BOX", dim=child1_dim, z=90.0)
        child1.add_layer(thk=10.0, material="comp1")
        child2 = Obj(type="BOX", dim=child2_dim, z=90.0)
        child2.add_layer(thk=10.0, material="comp2")
        main.add_child(child1)
        main.add_child(child2)

        mesher = OptimalMesh25D()
        mesher.set_pattern_obj(main, element_size=10.0, ratio=0.2)
        mesher.mesh_checkerboard()
        dragger = mesher.build([[
            {
                "z": 0.0,
                "element_size": 10.0,
                "areas": [{
                    "type": "BOX",
                    "dim": main_dim,
                    "material": "COMP1",
                }],
            },
            {
                "z": 90.0,
                "element_size": 10.0,
                "areas": [
                    {
                        "type": "BOX",
                        "dim": child1_dim,
                        "material": "COMP2",
                    },
                    {
                        "type": "BOX",
                        "dim": child2_dim,
                        "material": "COMP3",
                    },
                ],
            },
            {"z": 100.0},
        ]], preserve_mesh2d=True)

        elements = dragger.elements[:dragger.element_num]
        nodes = dragger.nodes[:dragger.node_num]
        components = dragger.element_comps[:dragger.element_num]
        hex_z = nodes[elements, 2]
        top_slab = (
            np.isclose(hex_z.min(axis=1), 90.0)
            & np.isclose(hex_z.max(axis=1), 100.0)
        )
        self.assertTrue(np.any(top_slab))

        bottom_xy = nodes[elements[top_slab, :4], :2]
        x = bottom_xy[:, :, 0]
        y = bottom_xy[:, :, 1]
        projected_areas = 0.5 * np.abs(
            np.sum(x * np.roll(y, -1, axis=1), axis=1)
            - np.sum(y * np.roll(x, -1, axis=1), axis=1)
        )
        top_components = components[top_slab]

        expected_areas = {
            "COMP1": 9801.0,
            "COMP2": 100.0,
            "COMP3": 99.0,
        }
        for material, expected_area in expected_areas.items():
            material_area = projected_areas[
                top_components == dragger.comps[material]
            ].sum()
            self.assertAlmostEqual(float(material_area), expected_area)
        self.assertAlmostEqual(float(projected_areas.sum()), 10000.0)

        lower_slab = (
            np.isclose(hex_z.min(axis=1), 80.0)
            & np.isclose(hex_z.max(axis=1), 90.0)
        )
        lower_top_node_ids = np.unique(elements[lower_slab, 4:])
        child_bottom_node_ids = np.unique(
            elements[
                top_slab
                & (components != dragger.comps["COMP1"]),
                :4,
            ]
        )
        np.testing.assert_array_equal(
            np.setdiff1d(child_bottom_node_ids, lower_top_node_ids),
            [],
        )

        main_hexes = elements[components == dragger.comps["COMP1"]]
        self.assertEqual(float(nodes[main_hexes, 2].max()), 100.0)

    def test_full_domain_capacity_proves_z_planes_before_allocating(self):
        mesher = self._integrity_mesher()
        stack = [[
            {
                "z": 1.0e16,
                "areas": [_area()],
                "element_size": 1.0,
            },
            {"z": 1.0e16 + 2.0},
        ]]

        with mock.patch.object(
            Dragger,
            "_pre_allocate_elements",
            side_effect=AssertionError("allocated elements before z proof"),
        ) as element_allocate, mock.patch.object(
            Dragger,
            "_pre_allocate_nodes",
            side_effect=AssertionError("allocated nodes before z proof"),
        ) as node_allocate:
            with self.assertRaisesRegex(ValueError, "float64 z resolution"):
                mesher.build(stack, preserve_mesh2d=True)

        element_allocate.assert_not_called()
        node_allocate.assert_not_called()

    def test_failed_build_rolls_back_material_volume_diagnostic(self):
        mesher = self._integrity_mesher()
        metal = {
            "type": "NORMAL",
            "material": "M1",
            "density": 0.0,
            "ranges": [{"type": "BOX", "dim": [0.0, 0.0, 3.0, 3.0]}],
        }
        first_area = _area()
        first_area["metals"] = [metal]
        stack = [[
            {"z": 0.0, "areas": [first_area], "element_size": 1.0},
            {"z": 1.0, "areas": [_area(), _area()], "element_size": 1.0},
            {"z": 2.0},
        ]]

        with self.assertRaisesRegex(ValueError, "areas overlap"):
            mesher.build(stack, preserve_mesh2d=True)

        self.assertNotIn("volumn", metal)

    def test_normal_selector_area_overflow_fails_closed(self):
        mesher = OptimalMesh25D()
        domain = [0.0, 0.0, 2.0e154, 1.0e154]
        mesher._set_pattern(
            [{
                "type": "BOX",
                "dim": domain,
                "bottom_z": 0.0,
                "top_z": 1.0,
            }],
            element_size=4.0e153,
            ratio=0.0,
            mesh_domain={"type": "BOX", "dim": domain},
        )
        mesher.mesh_checkerboard()
        metal = {
            "type": "NORMAL",
            "material": "M1",
            "density": 50.0,
            "ranges": [{"type": "BOX", "dim": domain}],
        }
        area = {
            "type": "BOX",
            "dim": domain,
            "material": "CORE",
            "metals": [metal],
        }

        with self.assertRaisesRegex(OverflowError, "total area"):
            mesher.build(
                [[
                    {"z": 0.0, "areas": [area], "element_size": 1.0},
                    {"z": 1.0},
                ]],
                preserve_mesh2d=True,
            )

        self.assertNotIn("volumn", metal)

    def test_reset_rejects_changed_structural_rail_metadata(self):
        mesher = self._integrity_mesher()
        mesher.rail_node_index["x"][0]["coord"] += 0.125

        with self.assertRaisesRegex(RuntimeError, "structural indices"):
            mesher.reset_snap_state()

    def test_build_rejects_area_boundary_that_cuts_through_a_mesh_cell(self):
        mesher = OptimalMesh25D()
        mesher._set_pattern(
            [{
                "type": "BOX",
                "dim": [0.0, 0.0, 10.0, 10.0],
                "bottom_z": 0.0,
                "top_z": 2.0,
            }],
            element_size=10.0,
            ratio=0.0,
            mesh_domain={"type": "BOX", "dim": [0.0, 0.0, 10.0, 10.0]},
        )
        mesh = mesher.mesh_checkerboard()
        baseline = mesh.nodes.copy()

        with self.assertRaisesRegex(ValueError, "not an active exact"):
            mesher.build(
                [[
                    {
                        "z": 0.0,
                        "areas": [{
                            "type": "BOX",
                            "dim": [0.0, 0.0, 5.0, 10.0],
                            "material": "CORE",
                        }],
                        "element_size": 1.0,
                    },
                    {"z": 1.0},
                ]]
            )

        np.testing.assert_array_equal(mesh.nodes, baseline)

    def test_build_rejects_unrepresented_hole_boundary(self):
        mesher = OptimalMesh25D()
        mesher._set_pattern(
            [{
                "type": "BOX",
                "dim": [0.0, 0.0, 10.0, 10.0],
                "bottom_z": 0.0,
                "top_z": 2.0,
            }],
            element_size=10.0,
            ratio=0.0,
            mesh_domain={"type": "BOX", "dim": [0.0, 0.0, 10.0, 10.0]},
        )
        mesher.mesh_checkerboard()
        area = {
            "type": "BOX",
            "dim": [0.0, 0.0, 10.0, 10.0],
            "material": "CORE",
            "holes": [{"type": "BOX", "dim": [2.0, 2.0, 8.0, 8.0]}],
        }

        with self.assertRaisesRegex(ValueError, "not an active exact"):
            mesher.build(
                [[
                    {"z": 0.0, "areas": [area], "element_size": 1.0},
                    {"z": 1.0},
                ]]
            )

    def test_build_boundary_must_exist_throughout_the_whole_slab(self):
        faces = [
            {
                "type": "LINE",
                "dim": [1.0, 0.0, 1.0, 1.0],
                "bottom_z": 0.0,
                "top_z": 1.0,
            },
            {
                "type": "LINE",
                "dim": [1.5, 0.0, 1.5, 1.0],
                "bottom_z": 2.0,
                "top_z": 3.0,
            },
        ]
        mesher = OptimalMesh25D()
        mesher._set_pattern(
            faces,
            element_size=1.0,
            ratio=1.0,
            mesh_domain={"type": "BOX", "dim": [0.0, 0.0, 3.0, 1.0]},
        )
        mesher.mesh_checkerboard()
        child_area = {
            "type": "BOX",
            "dim": [0.0, 0.0, 1.0, 1.0],
            "material": "CHILD",
        }

        with self.assertRaisesRegex(ValueError, "throughout z"):
            mesher.build(
                [[
                    {"z": 1.0, "areas": [child_area], "element_size": 1.0},
                    {"z": 2.0},
                ]]
            )

    def test_build_rejects_baseline_rail_at_displaced_span_endpoint(self):
        """A moved closed-span endpoint cannot masquerade as a safe rail."""
        faces = [
            {
                "type": "LINE",
                "dim": [1.0, 0.0, 1.0, 1.0],
                "bottom_z": 0.0,
                "top_z": 1.0,
            },
            {
                "type": "LINE",
                "dim": [1.5, 1.0, 1.5, 2.0],
                "bottom_z": 2.0,
                "top_z": 3.0,
            },
            {
                "type": "LINE",
                "dim": [0.0, 1.0, 3.0, 1.0],
                "bottom_z": 0.0,
                "top_z": 1.0,
            },
            {
                "type": "LINE",
                "dim": [0.0, 2.0, 3.0, 2.0],
                "bottom_z": 0.0,
                "top_z": 1.0,
            },
        ]
        mesher = OptimalMesh25D()
        mesher._set_pattern(
            faces,
            element_size=1.0,
            ratio=1.0,
            mesh_domain={"type": "BOX", "dim": [0.0, 0.0, 3.0, 3.0]},
        )
        mesh = mesher.mesh_checkerboard()
        baseline = mesh.nodes.copy()
        self.assertIsNone(mesher.rail_optimization_fallback)
        self.assertEqual(mesher.rails["x"][0]["coord"], 1.25)

        with self.assertRaisesRegex(ValueError, "throughout z"):
            mesher.build(
                [[
                    {
                        "z": 0.0,
                        "areas": [{
                            "type": "BOX",
                            "dim": [1.25, 1.0, 3.0, 2.0],
                            "material": "CORE",
                        }],
                        "element_size": 1.0,
                    },
                    {"z": 1.0},
                ]]
            )

        np.testing.assert_array_equal(mesh.nodes, baseline)

    def test_build_applies_snap_state_to_the_final_sentinel_plane(self):
        faces = [
            {
                "type": "LINE",
                "dim": [1.0, 0.0, 1.0, 1.0],
                "bottom_z": 1.0,
                "top_z": 2.0,
            },
            {
                "type": "LINE",
                "dim": [1.5, 0.0, 1.5, 1.0],
                "bottom_z": 3.0,
                "top_z": 4.0,
            },
        ]
        mesher = OptimalMesh25D()
        mesher._set_pattern(
            faces,
            element_size=1.0,
            ratio=1.0,
            mesh_domain={"type": "BOX", "dim": [0.0, 0.0, 3.0, 1.0]},
        )
        mesher.mesh_checkerboard()
        domain_area = {
            "type": "BOX",
            "dim": [0.0, 0.0, 3.0, 1.0],
            "material": "CORE",
        }

        dragger = mesher.build(
            [[
                {"z": 0.0, "areas": [domain_area], "element_size": 1.0},
                {"z": 1.0},
            ]],
            preserve_mesh2d=True,
        )
        top_nodes = dragger.nodes[:dragger.node_num]
        top_nodes = top_nodes[top_nodes[:, 2] == 1.0]

        self.assertTrue(np.any(top_nodes[:, 0] == 1.0))
        self.assertFalse(np.any(top_nodes[:, 0] == 1.25))

    def test_multiple_stacks_require_explicit_independent_body_opt_in(self):
        mesher = _build_zero_copy_mesher()
        stack = [
            {"z": 0.0, "areas": [_area()], "element_size": 1.0},
            {"z": 10.0},
        ]

        with self.assertRaisesRegex(ValueError, "non-conformal"):
            mesher.build([stack, stack], preserve_mesh2d=True)

    def test_build_rejects_a_stack_that_skips_pattern_z_events(self):
        """A slab may not cross a feature start or end without a boundary."""
        faces = [
            {
                "type": "LINE",
                "dim": [1.0, 0.0, 1.0, 3.0],
                "bottom_z": 0.0,
                "top_z": 4.0,
            },
            {
                "type": "LINE",
                "dim": [1.5, 0.0, 1.5, 3.0],
                "bottom_z": 6.0,
                "top_z": 10.0,
            },
        ]
        mesher = OptimalMesh25D()
        mesher._set_pattern(
            faces,
            element_size=5.0,
            ratio=0.2,
            mesh_domain={"type": "BOX", "dim": [0.0, 0.0, 3.0, 3.0]},
        )
        mesh = mesher.mesh_checkerboard()
        baseline = mesh.nodes.copy()

        with self.assertRaisesRegex(ValueError, "pattern z events"):
            mesher.build(
                [[
                    {"z": 0.0, "areas": [_area()], "element_size": 10.0},
                    {"z": 10.0},
                ]]
            )

        np.testing.assert_array_equal(mesh.nodes, baseline)

    def test_failed_build_restores_the_structural_2d_baseline(self):
        """A late layer error must not leave the zero-copy mesh snapped."""
        mesher = _build_zero_copy_mesher()
        baseline = mesher.mesh2d.nodes.copy()

        with self.assertRaisesRegex(ValueError, "element_size"):
            mesher.build(
                [[
                    {"z": 0.0, "areas": [_area()], "element_size": 10.0},
                    {"z": 10.0, "areas": [_area()], "element_size": 0.0},
                    {"z": 20.0},
                ]]
            )

        np.testing.assert_array_equal(mesher.mesh2d.nodes, baseline)

    def test_build_returns_dragger_and_mutates_mesh2d_by_default(self):
        mesher = _build_zero_copy_mesher()
        obj_list = [
            [
                {"z": 0, "areas": [_area()], "element_size": 10},
                {"z": 10},
            ]
        ]

        dragger = mesher.build(obj_list)

        self.assertIsInstance(dragger, Dragger)
        self.assertIs(mesher.dragger, dragger)
        self.assertTrue(np.shares_memory(dragger.node_2D, mesher.mesh2d.nodes))

        x_values = {
            round(float(x), 6)
            for x in mesher.mesh2d.nodes[:, 0]
        }
        self.assertIn(1.0, x_values)
        self.assertNotIn(1.25, x_values)

    def test_build_can_preserve_mesh2d_with_one_node_copy(self):
        mesher = _build_zero_copy_mesher()
        before = mesher.mesh2d.nodes.copy()
        obj_list = [
            [
                {"z": 0, "areas": [_area()], "element_size": 10},
                {"z": 10},
            ]
        ]

        dragger = mesher.build(obj_list, preserve_mesh2d=True)

        self.assertFalse(np.shares_memory(dragger.node_2D, mesher.mesh2d.nodes))
        np.testing.assert_allclose(mesher.mesh2d.nodes, before)
        x_values = {
            round(float(x), 6)
            for x in dragger.node_2D[:, 0]
        }
        self.assertIn(1.0, x_values)
        self.assertNotIn(1.25, x_values)

    def test_preserve_mesh2d_keeps_an_existing_active_snap_state(self):
        mesher = _build_zero_copy_mesher()
        mesher.apply_snap_rules_at_z(0.0)
        before = mesher.mesh2d.nodes.copy()
        managed_before = {
            axis: node_ids.copy()
            for axis, node_ids in mesher._current_changed_node_ids.items()
        }

        mesher.build(
            [[
                {"z": 0.0, "areas": [_area()], "element_size": 10.0},
                {"z": 10.0},
            ]],
            preserve_mesh2d=True,
        )

        np.testing.assert_array_equal(mesher.mesh2d.nodes, before)
        for axis in ("x", "y"):
            np.testing.assert_array_equal(
                mesher._current_changed_node_ids[axis],
                managed_before[axis],
            )
        self.assertIs(mesher._snap_state_nodes, mesher.mesh2d.nodes)

    def test_failed_preserved_build_restores_existing_active_snap_state(self):
        mesher = _build_zero_copy_mesher()
        mesher.apply_snap_rules_at_z(0.0)
        before = mesher.mesh2d.nodes.copy()
        invalid_area = _area()
        del invalid_area["material"]

        with self.assertRaisesRegex(ValueError, "material must"):
            mesher.build(
                [[
                    {"z": 0.0, "areas": [_area()], "element_size": 10.0},
                    {
                        "z": 10.0,
                        "areas": [invalid_area],
                        "element_size": 10.0,
                    },
                    {"z": 20.0},
                ]],
                preserve_mesh2d=True,
            )

        np.testing.assert_array_equal(mesher.mesh2d.nodes, before)
        self.assertIs(mesher._snap_state_nodes, mesher.mesh2d.nodes)

    def test_preserved_state_survives_dragger_constructor_failure(self):
        mesher = _build_zero_copy_mesher()
        mesher.apply_snap_rules_at_z(0.0)
        before = mesher.mesh2d.nodes.copy()
        ids_before = {
            axis: values.copy()
            for axis, values in mesher._current_changed_node_ids.items()
        }
        values_before = {
            axis: values.copy()
            for axis, values in mesher._current_changed_node_values.items()
        }

        with mock.patch(
            "optimal_checkerboard.mesher.Dragger",
            side_effect=MemoryError("synthetic allocation failure"),
        ):
            with self.assertRaises(MemoryError):
                mesher.build(
                    [[
                        {"z": 0.0, "areas": [_area()], "element_size": 10.0},
                        {"z": 10.0},
                    ]],
                    preserve_mesh2d=True,
                )

        np.testing.assert_array_equal(mesher.mesh2d.nodes, before)
        for axis in ("x", "y"):
            np.testing.assert_array_equal(
                mesher._current_changed_node_ids[axis],
                ids_before[axis],
            )
            np.testing.assert_array_equal(
                mesher._current_changed_node_values[axis],
                values_before[axis],
            )
        self.assertIs(mesher._snap_state_nodes, mesher.mesh2d.nodes)

    def test_build_syncs_snapped_top_layer_3d_nodes(self):
        faces = [
            {
                "type": "LINE",
                "dim": [1, 0, 1, 3],
                "bottom_z": 0,
                "top_z": 10,
            },
            {
                "type": "LINE",
                "dim": [1.5, 0, 1.5, 3],
                "bottom_z": 11,
                "top_z": 20,
            },
        ]
        mesher = OptimalMesh25D()
        mesher._set_pattern(
            faces,
            element_size=5,
            ratio=0.2,
            mesh_domain={"type": "BOX", "dim": [0, 0, 3, 3]},
        )
        mesher.mesh_checkerboard()
        area = _area()
        obj_list = [
            [
                {"z": 0, "areas": [area], "element_size": 10},
                {"z": 10, "areas": [area], "element_size": 1},
                {"z": 11, "areas": [area], "element_size": 10},
                {"z": 20},
            ]
        ]

        dragger = mesher.build(obj_list)

        nodes = dragger.nodes[:dragger.node_num]
        z11_nodes = nodes[np.isclose(nodes[:, 2], 11)]
        z11_x_values = {
            round(float(x), 6)
            for x in z11_nodes[:, 0]
        }
        self.assertIn(1.5, z11_x_values)
        self.assertNotIn(1.0, z11_x_values)
        self.assertNotIn(1.25, z11_x_values)

    def test_build_applies_distinct_raw_high_precision_z_events(self):
        """Verify build resolves raw z values to separate exact buckets."""
        first_z = 1.23444
        second_z = 1.23446
        faces = [
            {
                "type": "LINE",
                "dim": [1, 0, 1, 5],
                "bottom_z": first_z,
                "top_z": second_z,
            },
            {
                "type": "LINE",
                "dim": [1.5, 6, 1.5, 11],
                "bottom_z": second_z,
                "top_z": second_z + 1,
            },
        ]
        mesher = OptimalMesh25D()
        mesher._set_pattern(
            faces,
            element_size=5,
            ratio=0.2,
            mesh_domain={"type": "BOX", "dim": [0, 0, 3, 11]},
        )
        mesher.mesh_checkerboard()
        area = {"type": "BOX", "dim": [0, 0, 3, 11], "material": "CORE"}

        dragger = mesher.build(
            [[
                {"z": first_z, "areas": [area], "element_size": 1},
                {"z": second_z, "areas": [area], "element_size": 1},
                {"z": second_z + 1},
            ]],
            preserve_mesh2d=True,
        )
        nodes = dragger.nodes[:dragger.node_num]
        first_plane = nodes[np.isclose(nodes[:, 2], first_z, atol=1e-9)]
        second_plane = nodes[np.isclose(nodes[:, 2], second_z, atol=1e-9)]

        self.assertTrue(
            np.any(
                np.isclose(first_plane[:, 0], 1.0)
                & (first_plane[:, 1] <= 5)
            )
        )
        self.assertTrue(
            np.any(
                np.isclose(second_plane[:, 0], 1.5)
                & (second_plane[:, 1] >= 6)
            )
        )

    def test_each_object_stack_starts_from_shared_rail_baseline(self):
        """Verify stacked traversals restore coordinates and element areas."""
        faces = [
            {
                "type": "LINE",
                "dim": [1, 0, 1, 3],
                "bottom_z": 10,
                "top_z": 20,
            },
            {
                "type": "LINE",
                "dim": [1.5, 0, 1.5, 3],
                "bottom_z": 30,
                "top_z": 40,
            },
        ]
        mesher = OptimalMesh25D()
        mesher._set_pattern(
            faces,
            element_size=5,
            ratio=0.2,
            mesh_domain={"type": "BOX", "dim": [0, 0, 3, 3]},
        )
        mesher.mesh_checkerboard()
        area = _area()
        first_stack = [
            {"z": z, "areas": [area], "element_size": 10}
            for z in (0, 10, 20, 30)
        ] + [{"z": 40}]
        metal = {
            "type": "NORMAL",
            "material": "M1",
            "density": 0,
            "ranges": [{"type": "BOX", "dim": [0, 0, 1.25, 3]}],
        }
        second_area = dict(area)
        second_area["metals"] = [metal]
        second_stack = [
            {"z": 0, "areas": [second_area], "element_size": 10},
            {"z": 5},
        ]

        dragger = mesher.build(
            [first_stack, second_stack],
            preserve_mesh2d=True,
            allow_independent_bodies=True,
        )
        nodes = dragger.nodes[:dragger.node_num]
        z0_x_values, counts = np.unique(
            np.round(nodes[np.isclose(nodes[:, 2], 0), 0], 6),
            return_counts=True,
        )
        z0_counts = dict(zip(z0_x_values, counts))

        self.assertEqual(z0_counts[1.25], 4)
        self.assertNotIn(1.5, z0_counts)
        self.assertAlmostEqual(metal["volumn"], 3.75)


if __name__ == "__main__":
    unittest.main()
