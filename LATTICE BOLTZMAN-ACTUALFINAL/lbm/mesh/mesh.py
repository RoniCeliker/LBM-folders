from __future__ import annotations
from typing import Union
import numpy as np
import numba as nb
from numba import cuda
import json
import time
import matplotlib.pyplot as plt

import ffa
from lbm.mesh.node import Node
from lbm.mesh.cell import Cell, compute_normal_cuda
from lbm.mesh.face import Face
from lbm.mesh.boundary import Boundary



def read_ffa_mesh(file_name: str) -> Mesh:
    mesh = Mesh()
    ffa_msh = ffa.read(file_name)
    coord = ffa_msh.get('region').get('coordinates')

    # Create nodes
    mesh.nodes = [Node(r=coord.data[i], index=i)
                  for i in range(coord.nsiz)]

    # Create cells
    hexa = ffa_msh.get('region').get('element_group').get('element_nodes')
    mesh.cells = [Cell(nodes=[mesh.nodes[j] for j in hexa.data[i] - 1], index=i)
                  for i in range(hexa.nsiz)]

    b_list = ffa_msh.get('region').getl('boundary')
    b_counter = 0
    for b in b_list:
        b_name = b.get('boundary_name').data[0, 0].strip()
        b_nodes = b.get('belem_group').get('bound_elem_nodes')
        mesh.boundaries[b_name] = Boundary(name=b_name, index=b_counter)
        faces = [Face(nodes=[mesh.nodes[j] for j in b_nodes.data[i] - 1], index=i, boundary=mesh.boundaries[b_name])
                 for i in range(b_nodes.nsiz)]
        mesh.boundaries[b_name].faces = faces
        b_counter += 1
    return mesh



def compute_all_cell_face_normals_init(cell_face_idx, cell_node_idx, node_coordinates):
    cell_face_idx_device = cuda.to_device(cell_face_idx)
    cell_node_idx_device = cuda.to_device(cell_node_idx)
    node_coordinates_device = cuda.to_device(node_coordinates)
    normals_device = cuda.device_array(
        (cell_node_idx.shape[0], cell_face_idx.shape[0] * node_coordinates.shape[1])
    )

    threads_per_block = (16, 16)
    blocks_per_grid = (
        (cell_node_idx.shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
        (cell_face_idx.shape[0] + threads_per_block[1] - 1) // threads_per_block[1],
    )

    compute_all_cell_face_normals_cuda[blocks_per_grid, threads_per_block](
        cell_face_idx_device, cell_node_idx_device, node_coordinates_device, normals_device
    )

    normals = normals_device.copy_to_host()
    return normals



@cuda.jit
def compute_all_cell_face_normals_cuda(cell_face_idx, cell_node_idx, node_coordinates, normals_device):
    """
    CUDA kernel to compute all cell face normals.

    Parameters:
    - cell_face_idx: Indices of nodes defining each face in a cell.
    - cell_node_idx: Indices of nodes in each cell.
    - node_coordinates: Coordinates of all nodes.
    - normals_device: Output array to store computed normals.
    """
    cell_id, face_id = cuda.grid(2)
    n_cells, n_faces = cell_node_idx.shape[0], cell_face_idx.shape[0]
    dim = node_coordinates.shape[1]  # 2 for 2D, 3 for 3D

    if cell_id < n_cells and face_id < n_faces:
        # Compute node indices for the current face of the current cell
        node_indices = cuda.local.array(4, dtype=nb.int32)  # Maximum 4 nodes per face
        for i in range(cell_face_idx.shape[1]):  # Iterate over the number of nodes per face
            node_indices[i] = cell_node_idx[cell_id, cell_face_idx[face_id, i]]

        # Preallocate flattened array for face coordinates
        face_coords = cuda.local.array(12, dtype=nb.float64)  # 4 nodes * 3D (or adjust for 2D)
        for i in range(len(node_indices)):
            for j in range(dim):
                face_coords[i * dim + j] = node_coordinates[node_indices[i], j]

        # Preallocate normal array
        n = cuda.local.array(3, dtype=nb.float64)  # 3D normal
        compute_normal_cuda(face_coords, dim, n)

        # Write the normal into the output device array
        for d in range(dim):
            normals_device[cell_id, face_id * dim + d] = n[d]


def compute_cardinal_index(cardinals, cell_face_idx, cell_node_idx, node_coordinates):
    normals_device = cuda.to_device(
        compute_all_cell_face_normals_init(cell_face_idx, cell_node_idx, node_coordinates)
    )
    cardinal_idx_device = cuda.device_array((cell_node_idx.shape[0], cell_face_idx.shape[0]), dtype=np.uint32)
    dim = node_coordinates.shape[1]
    n_cells = cell_node_idx.shape[0]
    n_faces = cell_face_idx.shape[0]

    threads_per_block = (16, 16)
    blocks_per_grid = (
        (n_cells + threads_per_block[0] - 1) // threads_per_block[0],
        (n_faces + threads_per_block[1] - 1) // threads_per_block[1],
    )

    compute_cardinal_index_cuda[blocks_per_grid, threads_per_block](
        cuda.to_device(cardinals), normals_device, cardinal_idx_device, dim, n_cells, n_faces
    )

    return cardinal_idx_device.copy_to_host()

@cuda.jit
def compute_cardinal_index_cuda(cardinals, normals_device, cardinal_idx_device, dim, n_cells, n_faces):
    cell_id, face_id = cuda.grid(2)
    if cell_id < n_cells and face_id < n_faces:
        # Extract the face normal
        start_idx = face_id * dim
        face_normal = cuda.local.array(shape=(3,), dtype=nb.float64)
        for d in range(dim):
            face_normal[d] = normals_device[cell_id, start_idx + d]
        
        # Find matching cardinal index
        for k in range(cardinals.shape[0]):
            match = True
            for d in range(dim):
                if cardinals[k, d] != face_normal[d]:
                    match = False
                    break
            if match:
                cardinal_idx_device[cell_id, face_id] = k
                break


class Mesh:
    def __init__(self):
        self._nodes = []
        self._cells = []
        self._boundaries = dict()

    @property
    def nodes(self) -> list[Node]:
        return self._nodes

    @nodes.setter
    def nodes(self, nodes):
        self._nodes = nodes

    @property
    def cells(self) -> list[Cell]:
        return self._cells

    @cells.setter
    def cells(self, cells):
        self._cells = cells

    @property
    def faces(self):
        return np.concatenate([self.boundaries[key].faces for key in self.boundaries])

    @property
    def n(self) -> int:
        return len(self.cells)

    @property
    def boundaries(self) -> dict:
        return self._boundaries

    @boundaries.setter
    def boundaries(self, boundaries: dict):
        self._boundaries = boundaries

    def write_ffa(self, mesh_file):
        head = ffa.FFA("unstr_grid_data")
        head.append(ffa.FFA("title", m_data="LBM Grid"))
        region = ffa.FFA("region")
        region.append(ffa.FFA("region_name", "volume_elements"))
        region.append(ffa.FFA("coordinates", np.array([node.r for node in self.nodes])))
        for key in self.boundaries:
            boundary = ffa.FFA("boundary")
            boundary.append(ffa.FFA("boundary_name", key))
            belem_group = ffa.FFA("belem_group")
            index = np.array([[node.index for node in face.nodes] for face in self.boundaries[key].faces],
                             dtype=np.int32)
            if index.shape[1] == 2:
                belem_group.append(ffa.FFA("bound_elem_type", f"{'bar2':72}"))
            if index.shape[1] == 4:
                belem_group.append(ffa.FFA("bound_elem_type", f"{'quad4':72}"))
            belem_group.append(ffa.FFA("bound_elem_nodes", index + 1))
            boundary.append(belem_group)
            region.append(boundary)
        element_group = ffa.FFA("element_group")
        index = np.array([[node.index for node in cell.nodes] for cell in self.cells],
                         dtype=np.int32)
        if index.shape[1] == 2:
            element_group.append(ffa.FFA("element_type", f"{'bar2':72}"))
        if index.shape[1] == 4:
            element_group.append(ffa.FFA("element_type", f"{'quad4':72}"))
        if index.shape[1] == 8:
            element_group.append(ffa.FFA("element_type", f"{'hexa8':72}"))
        element_group.append(ffa.FFA("element_nodes", index + 1))
        region.append(element_group)
        head.append(region)
        head.write(mesh_file)

    def write_ffa_connectivity(self, file_name: str):
        head = ffa.FFA("grid_con_data")
        region = ffa.FFA("region")
        for key in self.boundaries:
            boundary = ffa.FFA("boundary")
            boundary.append(ffa.FFA("boundary_name", key))
            data = np.array([[face.neighbours[0].index, face.neighbours[0].neighbours.index(face)]
                             for face in self.boundaries[key].faces], dtype=np.int64)
            boundary.append(ffa.FFA("cell_idx", data[:, 0]))
            boundary.append(ffa.FFA("card_idx", data[:, 1]))
            region.append(boundary)
        data = np.array([[neighbour.index if isinstance(neighbour, Cell) else cell.index
                          for neighbour in cell.neighbours]
                         for cell in self.cells], dtype=np.int64)
        cell_group = ffa.FFA("volume")
        cell_group.append(ffa.FFA("cell_idx", data))
        region.append(cell_group)
        head.append(region)
        head.write(file_name)

    def read_ffa_connectivity(self, file_name: str):
        head = ffa.read(file_name)
        region = head.get("region")

        cell_group = region.get("volume")
        cell_idx = cell_group.get("cell_idx").data
        for cell, idx in zip(self.cells, cell_idx):
            cell.neighbours = [self.cells[i] for i in idx]

        boundary_list = region.getl("boundary")
        for boundary in boundary_list:
            key = boundary.get("boundary_name").data[0, 0]
            if key in self.boundaries:
                cell_idx = boundary.get("cell_idx").data
                card_idx = boundary.get("card_idx").data
                for face, idx1, idx2 in zip(self.boundaries[key].faces, cell_idx, card_idx):
                    face.neighbours = [self.cells[idx1[0]]]
                    self.cells[idx1[0]].neighbours[idx2[0]] = face

    def renumber_nodes(self):
        for i in range(len(self.nodes)):
            self.nodes[i].index = i

    def renumber_cells(self):
        for i in range(len(self.cells)):
            self.cells[i].index = i

    def find_node(self, r) -> Union[Node, None]:
        r = np.array(r, np.float64)
        r0 = np.array([node.r for node in self.nodes])
        idx = np.where(np.linalg.norm(r0 - r, axis=1) < 1e-3)[0]
        if idx.size > 0:
            return self.nodes[idx[0]]
        else:
            return None

    def find_face(self, r) -> Union[Face, None]:
        for key in self.boundaries:
            face = self.boundaries[key].find_face(r)
            if face is not None:
                return face
        return None

    def find_cell(self, r) -> Union[Cell, None]:
        r = np.array(r, np.float64)
        r0 = np.array([cell.coordinate for cell in self.cells])
        idx = np.where(np.linalg.norm(r0 - r, axis=1) < 1e-3)[0]
        if idx.size > 0:
            return self.cells[idx[0]]
        else:
            return None

    def setup(self):
        print('__add_elements_to_nodes')
        t0 = time.perf_counter()
        self.__add_elements_to_nodes()
        print(f"Time taken: {time.perf_counter() - t0}\n")

        print('__create_cell_connectivity')
        t0 = time.perf_counter()
        self.__create_cell_connectivity()
        print(f"Time taken: {time.perf_counter() - t0}\n")

        print('__create_face_connectivity')
        t0 = time.perf_counter()
        self.__create_face_connectivity()
        print(f"Time taken: {time.perf_counter() - t0}\n")

        # Clear node elements to clean up some memory
        self.__clear_node_elements()

    def setup_bc(self, bc_file: Union[str, dict]):
        if isinstance(bc_file, str):
            with open(bc_file) as f:
                d = json.load(f)
                for bc in d['boundary_conditions']:
                    self.boundaries[bc['boundary_name']].set_bc(bc)
        elif isinstance(bc_file, dict):
            for bc in bc_file['boundary_conditions']:
                self.boundaries[bc['boundary_name']].set_bc(bc)

    def plot_mesh_2d(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        for cell in self.cells:
            plot_cell_2d(ax, cell)
        for key in self.boundaries:
            for face in self.boundaries[key].faces:
                plot_face_2d(ax, face, key)
        for node in self.nodes:
            plot_node_2d(ax, node)
        plt.show()

    def __add_elements_to_nodes(self):
        # Add elements (cells/faces) to all nodes
        for cell in self.cells:
            [node.append_element(cell) for node in cell.nodes]
        for key in self.boundaries:
            for face in self.boundaries[key].faces:
                [node.append_element(face) for node in face.nodes]

    def __clear_node_elements(self):
        [node.clear() for node in self.nodes]

    def __create_cell_connectivity(self):
        # Create connectivity between cells/faces so we can traverse the grid
        #[cell.create_connectivity() for cell in self.cells]

        t0 = time.perf_counter()
        node_idx = np.array([[node.index for node in cell.nodes] for cell in self.cells])
        node_coordinate = np.array([node.r for node in self.nodes])
        cell_face = self.cells[0].get_cell_face()
        cardinals = self.cells[0].get_cardinals()
        cardinal_idx = compute_cardinal_index(cardinals, cell_face, node_idx, node_coordinate)
        print(f"\tcardinal_idx: {time.perf_counter() - t0}")

        t0 = time.perf_counter()
        [cell.create_connectivity2(c_idx) for cell, c_idx in zip(self.cells, cardinal_idx)]
        print(f"\tcreate_connectivity: {time.perf_counter() - t0}")

    def __create_face_connectivity(self):
        # Create connectivity between cells/faces so we can traverse the grid
        for key in self.boundaries:
            [face.create_connectivity() for face in self.boundaries[key].faces]
            #[self.boundaries[key].delete_face(face) for face in self.boundaries[key].faces[::-1]
            # if face.neighbours is None]


def plot_node_2d(ax, node):  # pragma: no cover
    r = node.r
    r = np.concatenate((r, [r[0]]), axis=0)
    ax.plot(r[0], r[1], marker='.', color=[0, 1, 0])
    plt.text(r[0], r[1], f'{node.index}', ha='center', va='center')


def plot_cell_2d(ax, cell):  # pragma: no cover
    r = np.array([node.r for node in cell.nodes])
    c = cell.coordinate
    r = np.concatenate((r, [r[0]]), axis=0)
    #dx = np.max(r[:, 0]) - np.min(r[:, 0])
    ax.plot(r[:, 0], r[:, 1], color=[0, 0, 0])
    plt.text(c[0], c[1], f'{cell.index}', ha='center', va='center')


def plot_face_2d(ax, face, key):  # pragma: no cover
    r = np.array([node.r for node in face.nodes])
    u = r[1] - r[0]
    c = face.coordinate
    n = face.normal
    r += 0.1*n
    c += 0.2*n
    ax.quiver(r[0, 0], r[0, 1], u[0], u[1],
              color=[0.0, 0.0, 1.0], angles='xy', scale=1, scale_units='xy',
              headwidth=3, headlength=5, minshaft=4)
    ax.quiver(c[0], c[1], n[0], n[1],
              color=[1.0, 0.0, 0.0], angles='xy', scale=2, scale_units='xy',
              headwidth=3, headlength=5, minshaft=3)
    plt.text(c[0], c[1], f'{face.index}', ha='center', va='center')
