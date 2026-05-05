import os
import sys
import unittest

SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
sys.path.insert(0, SRC_ROOT)

from optimal_checkerboard.algorithms.group_lines_by_z import _group_lines_by_z


class TestGroupLinesByZ(unittest.TestCase):
    def test_groups_by_rounded_z_and_sorts_keys(self):
        """Verify z grouping is rounded and returned in z order."""
        z2_line = [[0, 0, 2.00004], [1, 0, 2.9]]
        z1_line = [[0, 0, 1.00004], [1, 0, 1.0]]
        z1_line_2 = [[0, 1, 1.00003], [1, 1, 1.0]]

        grouped = _group_lines_by_z([z2_line, z1_line, z1_line_2])

        self.assertEqual(list(grouped), [1.0, 2.0])
        self.assertEqual(grouped[1.0], [z1_line, z1_line_2])
        self.assertEqual(grouped[2.0], [z2_line])


if __name__ == "__main__":
    unittest.main()
