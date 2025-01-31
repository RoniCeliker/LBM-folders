import numpy as np
from numba import cuda
from lbm.core.bc.boundary_condition import BoundaryCondition
from lbm.core.bc import BoundaryType, BoundaryClass
from lbm.core.stencil import Stencil
from lbm.mesh.cell import Cell
from lbm.mesh.face import Face
import lbm.core.nb_routines as nb_routines


class AntiBounceBack(BoundaryCondition):
    def __init__(self, name='AntiBounceBack', density=np.float32(1.0), capacity=0):
        super().__init__(name, BoundaryType.ANTI_BOUNCE_BACK, BoundaryClass.EXTERNAL)
        self.streaming_map = np.empty((capacity, 4), dtype=np.uint32)
        self.density = np.float32(density)
        self.wall_index = np.zeros((3, 0), dtype=np.uint32)
        ## arrays sent to device
        self.streaming_map_device = cuda.to_device(self.streaming_map)
        

    def streaming(self, lattice, stencil):
        nb_routines.boundary_anti_bounce_back(lattice.f_device, lattice.f0_device, lattice.u_device,
                                              self.streaming_map_device, self.density,
                                              lattice.w_device, lattice.c_device,
                                              stencil.inv_cs_2, stencil.inv_cs_4)

    def setup_streaming(self, pop_idx, cell: Cell, faces: list[Face], stencil):
        normal = -np.sum(np.array([face.normal for face in faces]), axis=0)
        cell_normal = cell.find_neighbour(normal)
        if not isinstance(cell_normal, Cell):
            cell_normal = cell
        streaming_map = np.array([cell.index, cell_normal.index, pop_idx, stencil.i_opp[pop_idx]],
                                 dtype=np.uint32)
        self.push_back(streaming_map)
        self.streaming_map_device = cuda.to_device(self.streaming_map[:self.size])

    def setup_boundary_data(self, stencil: Stencil):
        pass
