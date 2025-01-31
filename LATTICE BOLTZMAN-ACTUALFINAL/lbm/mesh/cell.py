from typing import Union
import numpy as np
import numba as nb
import math as math
from lbm.core.stencil import Stencil
from numba import cuda
from lbm.mesh.element import Element


@cuda.jit
def compute_normal_cuda(face_coords, dim, n):
    """
    Compute the normal for a face using CUDA device function.

    Parameters:
    - face_coords: Flattened array of node coordinates.
    - dim: Dimensionality (2 for 2D, 3 for 3D).
    - n: Output array to store the computed normal.
    """
    if dim == 3:  # 3D case
        v1 = cuda.local.array((3,), dtype=nb.float32)
        v2 = cuda.local.array((3,), dtype=nb.float32)
        for i in range(3):
            v1[i] = face_coords[3 * 3 + i] - face_coords[1 * 3 + i]
            v2[i] = face_coords[2 * 3 + i] - face_coords[0 * 3 + i]

        n[0] = v1[1] * v2[2] - v1[2] * v2[1]
        n[1] = v1[2] * v2[0] - v1[0] * v2[2]
        n[2] = v1[0] * v2[1] - v1[1] * v2[0]

    elif dim == 2:  # 2D case
        v = cuda.local.array((2,), dtype=nb.float32)
        for i in range(2):
            v[i] = face_coords[0 * 2 + i] - face_coords[1 * 2 + i]
        n[0] = -v[1]
        n[1] = v[0]

    # Normalize the vector
    norm = 0.0
    for i in range(dim):
        norm += n[i] ** 2
    norm = math.sqrt(norm)
    for i in range(dim):
        n[i] /= norm


""" def nb_parse_direction(direction: np.ndarray, cardinals: np.ndarray):
    # Dimensions
    n = direction.shape[0]  # Number of directions
    d = cardinals.shape[1]  # Dimensionality (2D or 3D)

    # Allocate device arrays
    direction_device = cuda.to_device(direction)  # Directions on device
    cardinals_device = cuda.to_device(cardinals)  # Cardinal directions on device
    index_list_device = cuda.device_array(n, dtype=np.int32)  # Output: indices
    direction_list_device = cuda.device_array((n, d), dtype=np.int32)  # Output: residual directions

    # Configure grid and block sizes
    threads_per_block = 128
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

    # Launch the CUDA kernel
    parse_direction_kernel[blocks_per_grid, threads_per_block](
        direction_device, cardinals_device, index_list_device, direction_list_device, n, d
    )
    cuda.synchronize()  # Ensure kernel execution is complete

    # Copy results back to host
    index_list = index_list_device.copy_to_host()
    direction_list = direction_list_device.copy_to_host()

    # Transform direction_list into a list of numpy arrays for easier handling
    direction_list_host = [direction_list[i] for i in range(n)]

    # Debugging output (optional, can be removed in production)
    print("Index List:", index_list)
    print("Direction List:", direction_list_host)

    return index_list, direction_list_host

@cuda.jit
def parse_direction_kernel(direction, cardinals, index_list, direction_list, n, d):
    tid = cuda.grid(1)  # Global thread ID

    if tid < n:  # Ensure thread ID is within bounds
        # Check if the entire direction row is zero
        zero_direction = True
        for k in range(d):
            if direction[tid, k] != 0:  # 2D indexing
                zero_direction = False
                break

        if zero_direction:
            index_list[tid] = -1  # Mark as invalid index
            for j in range(d):
                direction_list[tid, j] = 0  # Set corresponding direction to 0
            return

        # Loop over cardinal directions to find a match
        for j in range(cardinals.shape[0]):
            match = True
            for k in range(d):
                # Manual implementation of np.sign()
                direction_sign = 0
                if direction[tid, k] > 0:
                    direction_sign = 1
                elif direction[tid, k] < 0:
                    direction_sign = -1

                cardinal_sign = 0
                if cardinals[j, k] > 0:
                    cardinal_sign = 1
                elif cardinals[j, k] < 0:
                    cardinal_sign = -1

                if cardinal_sign != direction_sign:
                    match = False
                    break

            if match:
                index_list[tid] = j  # Save the matched index
                # Compute the new direction
                for k in range(d):
                    direction_list[tid, k] = direction[tid, k] - cardinals[j, k]
                return

        # If no match is found, set invalid index
        index_list[tid] = -1 """

@nb.njit
def nb_parse_direction(direction: np.ndarray, cardinals) -> (list[int], list[np.ndarray]):
    index_list = []
    direction_list = []
    for i in range(direction.size):
        if direction[i] == 0:
            continue
        # Find the index of the cardinal direction
        idx = np.nonzero(cardinals[:, i] == np.sign(direction[i]))[0][0]
        index_list.append(idx)
        # Directions should move towards 0
        direction_list.append(direction - cardinals[idx])
    return index_list, direction_list


class Cell(Element):
    __cardinals = [np.array([[-1], [1]], dtype=np.int32),  # 1D
                   np.array([[-1, 0], [1, 0],
                             [0, -1], [0, 1]], dtype=np.int32),  # 2D
                   np.array([[-1, 0, 0], [1, 0, 0],
                             [0, -1, 0], [0, 1, 0],
                             [0, 0, -1], [0, 0, 1]], dtype=np.int32)]  # 3D
    __cell_face_index = [np.array([[0], [1]], dtype=np.int32),  # 1D
                         np.array([[0, 1], [1, 2],
                                   [3, 0], [2, 3]], dtype=np.int32),  # 2D
                         np.array([[0, 3, 7, 4], [2, 1, 5, 6], [0, 4, 5, 1],
                                   [2, 6, 7, 3], [0, 1, 2, 3], [7, 6, 5, 4]], dtype=np.int32)]  # 3D

    def __init__(self, nodes=None, index: int = 0):
        super().__init__(nodes=nodes, index=index)

    def __str__(self):
        return f"<Cell:{self.index}>"

    def __repr__(self):
        return f"<Cell:{self.index}>"

    def set_cardinal_neighbour(self, cardinal, element) -> None:
        cardinals = self.__cardinals[self.cardinal_index()]
        idx = np.argmin(np.linalg.norm(cardinal - cardinals, axis=1))
        self.neighbours[idx] = element

    def get_cell_face(self):
        return self.__cell_face_index[self.cardinal_index()]

    def get_cardinals(self):
        return self.__cardinals[self.cardinal_index()]

    @staticmethod
    def __get_face_normal(cell_nodes):  # OBSOLETE if numba is available
        if len(cell_nodes) == 4:
            n = np.cross(cell_nodes[3].r - cell_nodes[1].r, cell_nodes[2].r - cell_nodes[0].r)
            return n/np.linalg.norm(n)
        elif len(cell_nodes) == 2:
            r = cell_nodes[0].r - cell_nodes[1].r
            n = np.array([-r[1], r[0]])
            return n/np.linalg.norm(n)

    def cardinal_index(self) -> int:
        if len(self.nodes) == 8:
            return 2
        elif len(self.nodes) == 4:
            return 1
        elif len(self.nodes) == 2:
            return 0
        else:
            raise Exception("Invalid cell.")

    def create_connectivity2(self, cardinal_idx) -> None:
        cardinal_index = self.cardinal_index()
        self.neighbours = [None] * len(cardinal_idx)

        node_idx = (0, -2,)
        for i in range(2):
            cells = [element for element in self.nodes[node_idx[i]].elements
                     if isinstance(element, Cell) and element != self]
            j = 0
            for cell_face_idx in self.__cell_face_index[cardinal_index][i::2]:
                nodes = [self.nodes[ii] for ii in cell_face_idx]
                for cell in cells:
                    if all(node in cell.nodes for node in nodes):
                        idx = cardinal_idx[j*2 + i]
                        self.neighbours[idx] = cell
                        break
                j += 1

    def create_connectivity(self) -> None:
        cardinal_index = self.cardinal_index()
        cardinals = self.__cardinals[cardinal_index]
        self.neighbours = [None] * len(cardinals)

        node_idx = (0, -2,)
        for i in range(2):
            cells = [element for element in self.nodes[node_idx[i]].elements
                     if isinstance(element, Cell) and element != self]
            # The order of the __cell_face_index is connected to the choice of node_idx
            #nodes = [[self.nodes[i] for i in idx] for idx in self.__cell_face_index[cardinal_index][i::2]]
            #for cell in cells:
            #    for node in nodes:
            #        if all(n in cell.nodes for n in node):
            #            n = self.__get_face_normal(node)
            #            idx = (n == cardinals).all(axis=1).nonzero()[0][0]
            #            self.neighbours[idx] = cell
            #            break

            for cell_face_idx in self.__cell_face_index[cardinal_index][i::2]:
                nodes = [self.nodes[i] for i in cell_face_idx]
                for cell in cells:
                    if all(node in cell.nodes for node in nodes):
                        n = compute_normal_cuda(np.array([node.r for node in nodes]))
                        #n = self.__get_face_normal(nodes)
                        idx = (n == cardinals).all(axis=1).nonzero()[0][0]
                        self.neighbours[idx] = cell
                        break

    def find_neighbour(self, direction: Union[list, tuple, np.ndarray]) -> Union[list[Element], Element]:
        """
        Depth first search looking for an element in a relative position to a cell.
        :param direction: Relative position to cell, [-1, 0] would be in the negative X-direction.
        :return: The cell if found, list of faces if they would interface the cell, [] if outside the domain.
        """
        direction = np.array(direction, dtype=np.int32)
        return self.__recursive_dfs_cell(direction, cardinals=self.__cardinals[self.cardinal_index()])
        #return self.__iterative_dfs_cell(direction, cardinals=self.__cardinals[self.cardinal_index()])

    def __iterative_dfs_cell(self, direction: np.ndarray, cardinals: np.ndarray) -> Union[list[Element], Element]:
        """
        Iterative depth first search
        """
        element_stack = [self]
        direction_stack = [direction]
        visited = []
        faces = []
        while element_stack:
            element = element_stack.pop()
            direction = direction_stack.pop()
            if isinstance(element, Cell):
                if (direction == 0).all():
                    return element  # If the norm of direction is 0, we are at the target
            else:
                if (direction == 0).all():
                    faces.append(element)  # If the norm of direction is 0, we reached the end but not a cell
                continue
            if element not in visited:
                visited.append(element)
                index_list, direction_list = nb_parse_direction(direction, cardinals)
                direction_stack += direction_list
                [element_stack.append(element.neighbours[idx]) for idx in index_list]
        return faces  # Since we did not find a target cell, return the faces that would be next to it

    def __recursive_dfs_cell(self, direction: np.ndarray, cardinals: np.ndarray,
                             visited: list = None, faces: list = None) -> Union[list[Element], Element]:
        """
        Recursive depth first search
        """
        if (direction == 0).all():
            return self  # If the norm of direction is 0, we are at the target
        if faces is None:
            faces = []
        if visited is None:
            visited = []
        visited.append(self)
        index_list, direction_list = nb_parse_direction(direction, cardinals)
        for index_i, direction_i in zip(index_list, direction_list):
            element = self.neighbours[index_i]
            if not isinstance(element, Cell):
                if (direction_i == 0).all():
                    faces.append(element)  # If the norm of direction is 0, we reached the end but not a cell
                continue
            if element not in visited:
                element = element.__recursive_dfs_cell(direction_i, cardinals=cardinals, visited=visited, faces=faces)
                if isinstance(element, Cell):
                    return element  # Target element found
        return faces  # Since we did not find a target cell, return the faces that would be next to it
