import numpy as np
from numba import cuda
from lbm.core.bc.boundary_condition import BoundaryCondition
from lbm.core.bc import BoundaryType, BoundaryClass
from lbm.core.stencil import Stencil
from lbm.mesh.cell import Cell
from lbm.mesh.face import Face
import lbm.core.nb_routines as nb_routines


class Symmetry(BoundaryCondition):
    def __init__(self, name='Symmetry', capacity=0):
        super().__init__(name, BoundaryType.SYMMETRY, BoundaryClass.WALL)
        self.streaming_map = np.empty((capacity, 4), dtype=np.uint32)
        self.streaming_map_device = cuda.to_device(self.streaming_map)

    def streaming(self, lattice, stencil):
        nb_routines.boundary_symmetry(lattice.f_device, lattice.f0_device, self.streaming_map_device)

    def setup_streaming(self, pop_idx, cell: Cell, faces: list[Face], stencil):
        normal = np.sum(np.array([face.normal for face in faces]), axis=0)
        element_n = cell.find_neighbour(normal)
        if element_n and isinstance(element_n, list):
            bc_n = element_n[0].boundary.bc
            if bc_n.boundary_type != self.boundary_type:
                # Delegate streaming
                bc_n.setup_streaming(pop_idx, cell, faces, stencil)
                return
        pop_idx_sym = stencil.i_sym[stencil.c_pop(normal), pop_idx]
        cell_sym = cell.find_neighbour(-(stencil.c[pop_idx] + normal))
        streaming_map = np.array([cell.index, cell_sym.index, pop_idx, pop_idx_sym],
                                 dtype=np.uint32)
        self.push_back(streaming_map)
        self.streaming_map_device = cuda.to_device(self.streaming_map[:self.size])

    def setup_boundary_data(self, stencil: Stencil):
        pass
