from typing import Union
from itertools import repeat
from numba import cuda
import numpy as np
import time
import multiprocessing as mp
import ffa
from lbm.core.stencil import Stencil
import lbm.core.nb_routines as nb_routines
from lbm.core.bc.boundary_condition import bc_priority
from lbm.mesh.mesh import Mesh, read_ffa_mesh
from lbm.mesh.cell import Cell


class Lattice:
    def __init__(self, stencil: Stencil, mesh):
        self._stencil = stencil
        self._mesh = mesh
        self._tau = 1.0  # Relaxation time
        self._omega = 1.0/self.tau
        self._omega_minus = 1.0
        self._lambda = np.float32(0.25)
        self.f = np.zeros((self.stencil.q, self.mesh.n), dtype=np.float32)
        self.f0 = np.zeros_like(self.f)
        self.S = np.zeros_like(self.f)  # Source term
        self.feq = np.zeros_like(self.f)  # Particle distribution equilibrium function
        self.u = np.zeros((self.mesh.n, self.stencil.d), dtype=np.float32)  # Velocities
        self.g = np.zeros_like(self.u)  # Force
        self.rho = np.ones(self.mesh.n, dtype=np.float32)  # Densities
        self.S_const = (1.0 - 1.0 / (2.0 * self.tau)) * self.stencil.w * self.stencil.inv_cs_2
        self.force_vector = np.zeros(self.stencil.d, dtype=np.float32)
        self._bulk_streaming_map = np.empty((0, 0), dtype=np.uint32)
        self._bound_streaming_map = np.empty((0, 0), dtype=np.uint32)
        self._bulk_cell_index = np.empty(0, dtype=np.uint32)
        self._bc_list = []
        #Arrays sent to device
        self.w_device = cuda.to_device(self.stencil.w)
        self.c_device = cuda.to_device(self.stencil.c)
        self.inv_cs_2_device = cuda.to_device(stencil.inv_cs_2)
        self.inv_cs_4_device = cuda.to_device(stencil.inv_cs_4)
        self.u_device = cuda.to_device(self.u)
        self.g_device = cuda.to_device(self.g)
        self.rho_device = cuda.to_device(self.rho)
        self.omega_device = cuda.to_device(self._omega)
        self.f_device = cuda.to_device(self.f)
        self.f0_device = cuda.to_device(self.f0)
        self.feq_device = cuda.to_device(self.feq)
        self.S_device = cuda.to_device(self.S)
        self.S_const_device = cuda.to_device(self.S_const)
        self.force_vector_device = cuda.to_device(self.force_vector)
 



    @property
    def stencil(self):
        return self._stencil

    @property
    def mesh(self):
        return self._mesh

    @property
    def omega(self):
        return self._omega

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, tau):
        self._tau = tau  # Relaxation time
        self._omega = 1.0/self.tau
        self._omega_minus = 1.0/(self._lambda/(self._tau - 0.5) + 0.5)
        self.S_const = (1.0 - 1.0 / (2.0 * self._tau)) * self.w_device * self.stencil.inv_cs_2

    def init_f(self):
        self.equilibrium()
        self.f[:] = self.feq[:]
        self.f0[:] = self.feq[:]
        self.macroscopic()

    def setup_streaming(self, mesh: str = None, connectivity: str = None,
                        bc: Union[str, dict] = None, processes: int = 1):
        """
        Setup streaming maps for internal cells, cells next to boundaries, and boundary condition streaming.
        Inputs are only required for parallel processing.
        :param mesh: Path to mesh file.
        :param connectivity: Path to connectivity file.
        :param bc: Path to boundary condition file, or a corresponding dictionary.
        :param processes: Number of processing cores.
        """
        cell_range = [np.uint32(len(self.mesh.cells) * i / processes) for i in range(processes)]
        if len(cell_range) < processes + 1:
            cell_range.append(np.uint32(len(self.mesh.cells)))
        cell_range = [np.uint32([cell_range[i], cell_range[i + 1]]) for i in range(processes)]

        print('__setup_streaming_maps')
        t0 = time.perf_counter()
        if processes == 1:
            stream_data = [_setup_streaming_maps(stencil=self.stencil, grid=self.mesh)]
        elif processes > 0:
            with mp.Pool(processes=processes) as pool:
                stream_data = pool.starmap(_setup_streaming_maps,
                                           zip(repeat(self.stencil), cell_range,
                                               repeat(mesh), repeat(connectivity), repeat(bc)))
        print(f"Time taken: {time.perf_counter() - t0}\n")

        print('concatenate streaming maps')
        t0 = time.perf_counter()
        self._bulk_streaming_map = np.concatenate(tuple([data["bulk_streaming_map"]
                                                         for data in stream_data]), axis=1)
        self._bulk_cell_index = np.concatenate(tuple([data["bulk_cell_index"]
                                                      for data in stream_data]), axis=0)
        self._bound_streaming_map = np.concatenate(tuple([data["bound_streaming_map"]
                                                          for data in stream_data]), axis=0)
        self._bulk_streaming_map_device = cuda.to_device(self._bulk_streaming_map)
        self._bound_streaming_map_device = cuda.to_device(self._bound_streaming_map)  
        self._bulk_cell_index_device = cuda.to_device(self._bulk_cell_index.astype(np.int32)) 
        for key in self.mesh.boundaries:
            bc = self.mesh.boundaries[key].bc
            bc.streaming_map = np.concatenate(tuple([data["boundaries"][key]
                                                     for data in stream_data]), axis=0)
        print(f"Time taken: {time.perf_counter() - t0}\n")

        for key in self.mesh.boundaries:
            bc = self.mesh.boundaries[key].bc
            bc.shrink_to_fit()
            bc.setup_boundary_data(self.stencil)
            if bc not in self._bc_list:
                self._bc_list.append(bc)

    def total_mass(self):
        return np.sum(self.rho)
    
    def swap_f(self) -> None:
        """
         AB-pattern, meaning we must swap f
        """
        self.f, self.f0 = self.f0, self.f

    def equilibrium(self):
        nb_routines.equilibrium(self.feq_device, self.u_device, self.rho_device,
                                self.stencil.q, self.w_device, self.c_device,
                                self.stencil.inv_cs_2, self.stencil.inv_cs_4, self.stencil.d)

    def collision(self):
        nb_routines.collision_bgk(self.f_device, self.feq_device, self.S_device, self.omega)
        #nb_routines.collision_trt(self.f, self.feq, self.S, self.stencil.i_opp, self.omega, self._omega_minus)

    def boundaries(self):
        for bc in self._bc_list:
            bc.streaming(self, self.stencil)

    def streaming(self):
        nb_routines.streaming_bulk(self.f_device, self.f0_device, self.stencil.q,
                                   self._bulk_streaming_map_device, self._bulk_cell_index_device)
        nb_routines.streaming_boundary(self.f_device, self.f0_device, self._bound_streaming_map_device)


    def macroscopic(self):
        nb_routines.density(self.f_device, self.rho_device)
        if np.linalg.norm(self.force_vector) > 0:
            nb_routines.force(self.g_device, self.rho_device, self.stencil.d, self.force_vector_device)
            nb_routines.force_velocity(self.f_device, self.rho_device, self.u_device, self.g_device,
                                       self.c_device, self.stencil.d)
        else:
            nb_routines.velocity(self.f_device, self.rho_device, self.u_device,
                                 self.c_device, self.stencil.d)
            
    def source_force(self):
        if np.linalg.norm(self.force_vector) > 0:
            nb_routines.source_force(self.S_device, self.S_const_device, self.u_device, self.g_device,
                                     self.c_device, self.stencil.q, self.stencil.inv_cs_2)

    def write_ffa_streaming_maps(self, file_name: str):
        head = ffa.FFA("streaming_maps")
        region = ffa.FFA("region")
        for key in self.mesh.boundaries:
            boundary = ffa.FFA("boundary")
            boundary.append(ffa.FFA("boundary_name", key))
            boundary.append(ffa.FFA("map", np.array(self.mesh.boundaries[key].bc.streaming_map,
                                                    dtype=np.int64)))
            region.append(boundary)
        volume = ffa.FFA("volume")
        volume.append(ffa.FFA("bulk_map", np.array(self._bulk_streaming_map, dtype=np.int64)))
        volume.append(ffa.FFA("bulk_index", np.array(self._bulk_cell_index, dtype=np.int64)))
        volume.append(ffa.FFA("bound_map", np.array(self._bound_streaming_map, dtype=np.int64)))
        region.append(volume)
        head.append(region)
        head.write(file_name)

    def read_ffa_streaming_maps(self, file_name: str):
        head = ffa.read(file_name)
        region = head.get("region")
        # Volume
        volume = region.get("volume")
        self._bulk_streaming_map = volume.get("bulk_map").data
        self._bulk_cell_index = volume.get("bulk_index").data.flatten()
        self._bound_streaming_map = volume.get("bound_map").data
        # Boundary
        boundary_list = region.getl("boundary")
        for boundary in boundary_list:
            key = boundary.get("boundary_name").data[0, 0]
            if key in self.mesh.boundaries:
                self.mesh.boundaries[key].bc.streaming_map = boundary.get("map").data


def _setup_streaming_maps(stencil: Stencil, cell_range: Union[list, np.ndarray] = None,
                          grid: Union[str, Mesh] = None, connectivity: str = None, bc: Union[str, dict] = None):
    if isinstance(grid, str):  # if multi-processed, lets read the grid for each process
        grid = read_ffa_mesh(grid)
        grid.read_ffa_connectivity(connectivity)
    if cell_range is None:
        cells = grid.cells
    else:
        cells = grid.cells[cell_range[0]:cell_range[1]]  # Limit which cells to process
    if bc is not None:
        grid.setup_bc(bc)

    bulk_streaming_map = np.zeros((stencil.q, len(cells)), dtype=np.uint32)
    bulk_cell_index = np.empty(len(cells), dtype=np.uint32)
    bound_streaming_map = np.zeros((np.uint32(0.1*len(cells)), 3), dtype=np.uint32)
    _size = np.uint32(0)
    _capacity = bound_streaming_map.shape[0]
    for key in grid.boundaries:
        bc = grid.boundaries[key].bc
        bc.reserve(np.uint32(0.1*len(cells)))

    c_int = np.int32(stencil.c)
    i_bulk = np.uint32(0)
    for cell in cells:
        elements = [cell.find_neighbour(-c_int[iq]) for iq in range(stencil.q)]
        if all(isinstance(element, Cell) for element in elements):
            streaming_map = np.array([element.index if isinstance(element, Cell) else cell.index
                                      for element in elements], dtype=np.uint32)
            bulk_streaming_map[:, i_bulk] = streaming_map
            bulk_cell_index[i_bulk] = cell.index
            i_bulk += np.uint32(1)
            continue
        pop_idx = 0
        for element in elements:
            if not element:  # element is []
                # convex corners, must select boundary condition here in someway
                c_tmp = np.zeros_like(c_int[pop_idx])
                faces = []
                for iq in range(stencil.d):
                    if c_int[pop_idx, iq] == 0:
                        continue
                    c_tmp[:] = 0
                    c_tmp[iq] = c_int[pop_idx, iq]
                    tmp_element = cell.find_neighbour(-c_tmp)
                    if isinstance(tmp_element, list):  # The list is either empty or a list of faces
                        faces += tmp_element
                bc = bc_priority([face.boundary.bc for face in faces])
                bc.setup_streaming(pop_idx, cell, faces, stencil)
            elif isinstance(element, Cell):
                streaming_map = np.array([cell.index, element.index, pop_idx],
                                         dtype=np.uint32)
                if _size >= _capacity:
                    _capacity = np.uint32(np.max([np.uint32(2) * _capacity, np.uint32(1)]))
                    bound_streaming_map.resize((_capacity, bound_streaming_map.shape[1]))
                bound_streaming_map[_size] = streaming_map
                _size += np.uint32(1)
            else:  # Element is a list of Face's
                # For concave corners the length of the list > 1
                bc = bc_priority([face.boundary.bc for face in element])
                bc.setup_streaming(pop_idx, cell, element, stencil)
            pop_idx += 1

    bound_streaming_map = bound_streaming_map[0:_size]
    bulk_streaming_map = bulk_streaming_map[:, 0:i_bulk]
    bulk_cell_index = bulk_cell_index[0:i_bulk]
    [grid.boundaries[key].bc.shrink_to_fit() for key in grid.boundaries]
    return {"bulk_streaming_map": bulk_streaming_map,
            "bulk_cell_index": bulk_cell_index,
            "bound_streaming_map": bound_streaming_map,
            "boundaries": {key: grid.boundaries[key].bc.streaming_map for key in grid.boundaries}}
