"""Layer data structure for stackup material thickness.

A Layer stores one material and its local thickness so objects can accumulate
their vertical build-up in order.
"""


class Layer:
    """Represent one material layer in an object stack."""

    def __init__(self, material, thk):
        self.material = material
        self.thk = thk

    def copy(self):
        layer_dup = Layer(self.material, self.thk)
        return layer_dup

    def dict(self):
        return {
            "material": self.material,
            "thk": self.thk,
        }

    def dict_abs(self):
        return {
            "material": self.material,
            "thk": self.thk,
        }
