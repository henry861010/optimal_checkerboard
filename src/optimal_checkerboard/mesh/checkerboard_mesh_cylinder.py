"""Placeholder for future cylindrical checkerboard mesh generation."""


def checkerboard_mesh_cylinder(domain, x_list, y_list, element_size):
    """Generate a checkerboard mesh for a cylindrical footprint.

    The public dispatcher already routes CYLINDER domains here so the future
    implementation can be added without changing ``OptimalMesh25D``.
    """
    raise NotImplementedError(
        "CYLINDER checkerboard mesh generation is not implemented yet"
    )
