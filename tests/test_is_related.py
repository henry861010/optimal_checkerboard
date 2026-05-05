import os
import sys
import unittest

SRC_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
sys.path.insert(0, SRC_ROOT)

from optimal_checkerboard.algorithms.is_related import _is_related


class TestIsRelated(unittest.TestCase):
    def test_relation_accepts_exact_endpoint_buffer(self):
        """Verify the exact longitudinal buffer boundary is inclusive."""
        line = [[0, 0, 0], [0, 10, 0]]

        self.assertTrue(
            _is_related(line, [[0.5, 11, 0], [0.5, 15, 0]], l=1.0, v_h=0)
        )
        self.assertTrue(
            _is_related(line, [[0.5, -5, 0], [0.5, -1, 0]], l=1.0, v_h=0)
        )

    def test_vertical_related_and_unrelated_cases(self):
        """Verify vertical related-line checks for nearby separated spans."""
        line = [[0, 0, 0], [0, 10, 0]]

        self.assertTrue(
            _is_related(line, [[0.5, 12, 0], [0.5, 15, 0]], l=1.0, v_h=0)
        )
        self.assertFalse(
            _is_related(line, [[0.5, 10, 0], [0.5, 15, 0]], l=1.0, v_h=0)
        )
        self.assertFalse(
            _is_related(line, [[0.5, 10.5, 0], [0.5, 15, 0]], l=1.0, v_h=0)
        )
        self.assertFalse(
            _is_related(line, [[0.5, 5, 0], [0.5, 15, 0]], l=1.0, v_h=0)
        )
        self.assertFalse(
            _is_related(line, [[5, 12, 0], [5, 15, 0]], l=1.0, v_h=0)
        )

    def test_horizontal_related_and_unrelated_cases(self):
        """Verify horizontal related-line checks for nearby separated spans."""
        line = [[0, 5, 0], [10, 5, 0]]

        self.assertTrue(
            _is_related(
                line,
                [[-10, 5.5, 0], [-2, 5.5, 0]],
                l=1.0,
                v_h=1,
            )
        )
        self.assertFalse(
            _is_related(
                line,
                [[10, 5.5, 0], [15, 5.5, 0]],
                l=1.0,
                v_h=1,
            )
        )
        self.assertFalse(
            _is_related(line, [[10.5, 5.5, 0], [15, 5.5, 0]], l=1.0, v_h=1)
        )
        self.assertFalse(
            _is_related(line, [[8, 5.5, 0], [15, 5.5, 0]], l=1.0, v_h=1)
        )

    def test_invalid_relation_inputs_return_false(self):
        """Verify invalid or overlapping relation inputs return False."""
        self.assertFalse(
            _is_related(
                [[0, 0, 0], [0, 10, 1]],
                [[1, 11, 0], [1, 15, 0]],
                l=1.0,
                v_h=0,
            )
        )
        self.assertFalse(
            _is_related(
                [[0, 0, 0], [0, 10, 0]],
                [[1, 11, 0], [5, 11, 0]],
                l=1.0,
                v_h=0,
            )
        )
        self.assertFalse(
            _is_related(
                [[0, 0, 0], [0, 10, 0]],
                [[0, 5, 0], [0, 15, 0]],
                l=1.0,
                v_h=0,
            )
        )


if __name__ == "__main__":
    unittest.main()
