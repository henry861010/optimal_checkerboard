"""Geometry preprocessing algorithms for the 2.5D checkerboard mesher."""

from optimal_checkerboard.algorithms.mesh_coverage import (
    validate_box_domain_partition,
    validate_mesh_coverage,
)

__all__ = ["validate_box_domain_partition", "validate_mesh_coverage"]
