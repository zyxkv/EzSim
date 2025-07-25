import taichi as ti

import ezsim
from ezsim.repr_base import RBC


@ti.data_oriented
class Material(RBC):
    """
    The base class of materials.

    Note
    ----
    This class should *not* be instantiated directly.
    """

    def __init__(self):
        self._uid = ezsim.UID()

    @property
    def uid(self):
        return self._uid

    @classmethod
    def _repr_type(cls):
        return f"<ezsim.{cls.__module__.split('.')[-2]}.{cls.__name__}>"
