from lbm.core.bc.boundary_condition import BoundaryCondition
from lbm.core.bc import BoundaryType, BoundaryClass
from lbm.core.stencil import Stencil


class Orphan(BoundaryCondition):
    def __init__(self, name='Orphan'):
        super().__init__(name, BoundaryType.ORPHAN, BoundaryClass.NONE)

    def streaming(self, lattice, stencil):
        pass

    def setup_streaming(self, pop_idx, cell, faces, stencil: Stencil):
        pass

    def setup_boundary_data(self, stencil: Stencil):
        pass
