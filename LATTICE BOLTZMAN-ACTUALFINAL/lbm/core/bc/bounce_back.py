import numpy as np
from lbm.core.bc.boundary_condition import BoundaryCondition
from lbm.core.bc import BoundaryType, BoundaryClass
from lbm.core.stencil import Stencil
from lbm.mesh.cell import Cell
from lbm.mesh.face import Face
from numba import cuda
import lbm.core.nb_routines as nb_routines


class BounceBack(BoundaryCondition):
    def __init__(self, name='BounceBack', velocity=None, density=np.float64(1.0), capacity=0):
        super().__init__(name, BoundaryType.BOUNCE_BACK, BoundaryClass.WALL)
        self.streaming_map = np.empty((capacity, 3), dtype=np.uint32)
        self.velocity = np.array(velocity, dtype=np.float64) if velocity is not None else np.zeros(2, dtype=np.float64)
        self.density = density
        self.wall_data_device = None  # Initialize wall_data_device to None
        self.streaming_map_device = None  # Initialize streaming_map_device to None
        if velocity is not None:
            self.boundary_type = BoundaryClass.EXTERNAL

    def __repr__(self):
        if self.velocity is None:
            return super().__repr__()
        else:
            return f"<{self.boundary_type}:TYPE:{self.boundary_type}:" \
                   f"{self.name},velocity={*self.velocity,},density={self.density}>"

    def streaming(self, lattice, stencil):
        if self.wall_data_device is None or self.streaming_map_device is None:
            self.setup_boundary_data(stencil)  # Ensure setup_boundary_data is called
        nb_routines.boundary_bounce_back(lattice.f_device, lattice.f0_device, self.streaming_map_device, self.wall_data_device)

    def setup_streaming(self, pop_idx, cell: Cell, faces: list[Face], stencil):
        streaming_map = np.array([cell.index, pop_idx, stencil.i_opp[pop_idx]],
                                 dtype=np.uint32)
        self.push_back(streaming_map)

    def setup_boundary_data(self, stencil: Stencil):
        self.wall_data = np.zeros(stencil.q, dtype=np.float64)  # Redefine wall_data as a NumPy array
        if self.velocity is not None:           
            stencil_w_device = cuda.to_device(stencil.w)
            stencil_c_device = cuda.to_device(stencil.c)
            self.streaming_map_device = cuda.to_device(self.streaming_map)  # Assign to self.streaming_map_device
            self.wall_data_device = cuda.to_device(self.wall_data)  # Assign to self.wall_data_device
            velocity_device = cuda.to_device(self.velocity)
            
            threads_per_block = 256
            blocks_per_grid = (stencil.q + (threads_per_block - 1)) // threads_per_block
            
            setup_boundary_data_kernel[blocks_per_grid, threads_per_block](
                self.wall_data_device, velocity_device, self.density, stencil_w_device, stencil_c_device, stencil.cs, stencil.q
            )
    
@cuda.jit
def setup_boundary_data_kernel(wall_data, velocity, density, stencil_w, stencil_c, stencil_cs, q):
    i = cuda.grid(1)
    if i < q:
        if velocity[0] is not None:
            wall_data[i] = -2 * stencil_w[i] * density * (stencil_c[i, 0] * velocity[0] + stencil_c[i, 1] * velocity[1]) / stencil_cs**2