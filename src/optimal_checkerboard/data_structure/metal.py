"""Metal data structure for vertical material spans.

A Metal stores a z-range, material transition metadata, and optional face
ranges or holes that limit where the metal applies within an object.
"""

from .face import Face


class Metal:
    """Represent a metal segment with local and absolute z coordinates."""

    def __init__(
        self,
        type,
        begin,
        end,
        material,
        material_o=None,
        ranges=None,
        holes=None,
    ):
        self.type: str = type
        self.begin: float = begin
        self.end: float = end
        self.material_o: str = material_o
        self.material: str = material
        self.ranges: list[Face] = list(ranges) if ranges is not None else []
        self.holes: list[Face] = list(holes) if holes is not None else []

        self.begin_abs = None
        self.end_abs = None

    def set_position_abs(self, x=0, y=0, z=0):
        self.begin_abs = self.begin + z
        self.end_abs = self.end + z

        for face_range in self.ranges:
            face_range.set_position_abs(x, y, z)

        for hole in self.holes:
            hole.set_position_abs(x, y, z)

    def copy(self):
        ranges_dup = [face_range.copy() for face_range in self.ranges]
        holes_dup = [hole.copy() for hole in self.holes]
        metal_dup = Metal(
            self.type,
            self.begin,
            self.end,
            self.material,
            self.material_o,
            ranges_dup,
            holes_dup,
        )
        metal_dup.begin_abs = self.begin_abs
        metal_dup.end_abs = self.end_abs
        return metal_dup

    def info(self, isAbs=False):
        if isAbs:
            return {
                "type": self.type,
                "begin": self.begin_abs,
                "end": self.end_abs,
                "material_o": self.material_o,
                "material": self.material,
                "ranges": [face_range.info(isAbs=True) for face_range in self.ranges],
                "holes": [hole.info(isAbs=True) for hole in self.holes],
            }
        else:
            return {
                "type": self.type,
                "begin": self.begin,
                "begin_abs": self.begin_abs,
                "end": self.end,
                "end_abs": self.end_abs,
                "material_o": self.material_o,
                "material": self.material,
                "ranges": [face_range.info() for face_range in self.ranges],
                "holes": [hole.info() for hole in self.holes],
            }
