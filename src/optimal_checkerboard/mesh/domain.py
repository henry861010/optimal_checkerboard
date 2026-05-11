"""Resolve checkerboard mesh domains from geometry footprints."""

from copy import deepcopy
from numbers import Real


def mesh_domain_from_obj(obj):
    """Return the root-footprint mesh domain for an absolute ``Obj``."""
    face = getattr(obj, "face", None)
    if face is None:
        raise ValueError("Obj must provide a root face")
    if face.dim_abs is None:
        raise ValueError(
            "Obj root face has no absolute coordinates. "
            "Call set_position_abs before resolving the mesh boundary."
        )
    return mesh_domain_from_face(face.type, face.dim_abs)


def mesh_domain_from_face(face_type, dim):
    """Return a normalized mesh domain dictionary from one face footprint."""
    face_type = _normalize_type(face_type)

    if face_type == "BOX":
        bounds = _box_bounds(dim)
        return {"type": "BOX", "dim": list(bounds), "bbox": list(bounds)}

    if face_type == "CYLINDER":
        cylinder_dim = _cylinder_dim(dim)
        cx, cy, radius = cylinder_dim
        return {
            "type": "CYLINDER",
            "dim": cylinder_dim,
            "bbox": [cx - radius, cy - radius, cx + radius, cy + radius],
        }

    if face_type == "POLYGON":
        bbox = _polygon_bounds(dim)
        return {"type": "POLYGON", "dim": deepcopy(dim), "bbox": bbox}

    raise ValueError(
        f"Unsupported root footprint type for checkerboard mesh: {face_type}"
    )


def normalize_mesh_domain(domain):
    """Validate and copy a mesh domain dictionary."""
    if domain is None:
        return None
    if not isinstance(domain, dict):
        raise ValueError("mesh_domain must be a dictionary")

    domain_type = _normalize_type(domain.get("type"))
    if domain_type == "BOX":
        dim = domain.get("dim", domain.get("bbox"))
    else:
        dim = domain.get("dim")

    if dim is None:
        raise ValueError(f"{domain_type} mesh_domain must provide dim")
    return mesh_domain_from_face(domain_type, dim)


def _normalize_type(face_type):
    if face_type is None:
        raise ValueError("mesh domain type is required")
    return str(face_type).upper()


def _box_bounds(dim):
    if len(dim) != 4:
        raise ValueError("BOX mesh domain dim must be [xmin, ymin, xmax, ymax]")
    xmin, ymin, xmax, ymax = [_number(value) for value in dim]
    bounds = [min(xmin, xmax), min(ymin, ymax), max(xmin, xmax), max(ymin, ymax)]
    _validate_positive_bounds(bounds, "BOX")
    return bounds


def _cylinder_dim(dim):
    if len(dim) != 3:
        raise ValueError("CYLINDER mesh domain dim must be [cx, cy, radius]")
    cx, cy, radius = [_number(value) for value in dim]
    if radius <= 0:
        raise ValueError("CYLINDER mesh domain radius must be positive")
    return [cx, cy, radius]


def _polygon_bounds(dim):
    points = []
    for loop in dim:
        for point in loop:
            if len(point) < 2:
                raise ValueError("POLYGON mesh domain points must contain x and y")
            points.append((_number(point[0]), _number(point[1])))

    if not points:
        raise ValueError("POLYGON mesh domain must contain at least one point")

    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    bounds = [min(x_values), min(y_values), max(x_values), max(y_values)]
    _validate_positive_bounds(bounds, "POLYGON")
    return bounds


def _validate_positive_bounds(bounds, domain_type):
    xmin, ymin, xmax, ymax = bounds
    if xmin >= xmax or ymin >= ymax:
        raise ValueError(f"{domain_type} mesh domain must have positive area")


def _number(value):
    if not isinstance(value, Real):
        raise ValueError("mesh domain coordinates must be numeric")
    return float(value)
