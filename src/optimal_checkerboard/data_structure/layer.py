"""Layer data structure for stackup material thickness.

A Layer stores one material and its local thickness so objects can accumulate
their vertical build-up in order.
"""

import math


class Layer:
    """Represent one material layer in an object stack."""

    def __init__(self, material, thk):
        self.material = material
        self.thk = thk

    @property
    def thk(self):
        return self._thk

    @thk.setter
    def thk(self, value):
        try:
            thickness = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Layer thickness must be finite and greater than zero"
            ) from exc
        if not math.isfinite(thickness) or thickness <= 0.0:
            raise ValueError(
                "Layer thickness must be finite and greater than zero"
            )
        self._thk = thickness

    def copy(self):
        layer_dup = Layer(self.material, self.thk)
        return layer_dup

    def info(self, isAbs=False):
        return {
            "material": self.material,
            "thk": self.thk,
        }
