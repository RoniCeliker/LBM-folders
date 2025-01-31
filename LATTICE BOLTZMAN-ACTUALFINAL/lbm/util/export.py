import numpy as np


class Export:
    def __init__(self, lattice=None, grid=None):
        if grid is None and lattice is not None:
            self._grid = lattice.mesh
        else:
            self._grid = grid
        self._lattice = lattice
        self._file = None
        self._cache = dict()

    @property
    def grid(self):
        return self._grid

    @property
    def lattice(self):
        return self._lattice

    def write_vtk(self, filename: str, write_data: bool = True, write_cell_info: bool = False):
        with open(filename, 'wb') as self._file:
            self.__write_header()
            self.__write_points()
            self.__write_cells()
            if write_data and self.lattice is not None:
                self.__write_data()
            if write_cell_info:
                self.__write_cell_info()

    def write_vtk_ascii(self, filename: str):
        with open(filename, 'w') as self._file:
            self.__write_header_ascii()
            self.__write_points_ascii()
            self.__write_cells_ascii()
            self.__write_data_ascii()

    def __write_header(self):
        self._file.write(b'# vtk DataFile Version 3.0\n')
        self._file.write(b'lbm solver data\n')
        self._file.write(b'BINARY\n')
        self._file.write(b'DATASET UNSTRUCTURED_GRID\n')

    def __write_points(self):
        points = self.grid.nodes
        self._file.write(b'POINTS ')
        self._file.write(f'{len(points)}'.encode('ascii'))
        self._file.write(b' float\n')
        if 'points' in self._cache:
            coordinate = self._cache['points']
        else:
            #coordinate = np.array([[node.x, node.y, node.z] for node in points], dtype=np.float32).flatten()
            coordinate = np.array([node.r for node in points], dtype=np.float32).flatten()
            if points[0].r.size == 2:
                # If 2D, insert z = 0 after each xy pair
                coordinate = np.insert(coordinate, np.arange(2, coordinate.size + 1, 2), 0.0)
            self._cache['points'] = coordinate
        if np.little_endian:
            coordinate.byteswap().tofile(self._file, sep="")
        else:
            coordinate.tofile(self._file, sep="")
        self._file.write(b'\n')

    def __write_cells(self):
        cells = self.grid.cells
        size = len(cells[0].nodes)
        #size = np.int32(2**self.lattice.stencil.d)
        #size = np.int32(2**3)
        self._file.write(b'CELLS ')
        self._file.write(f'{len(cells)} {int((size + 1) * len(cells))}'.encode('ascii'))
        self._file.write(b'\n')
        if 'connectivity' in self._cache:
            nodes_idx = self._cache['connectivity']
        else:
            nodes = np.array([node for node in [cell.nodes for cell in cells]])
            nodes_idx = np.array([node.index for node in nodes.flatten()], dtype=np.int32)
            nodes_idx = np.insert(nodes_idx, np.arange(size, nodes_idx.size, size), size)
            nodes_idx = np.insert(nodes_idx, 0, size)
            self._cache['connectivity'] = nodes_idx
        if np.little_endian:
            nodes_idx.byteswap().tofile(self._file, sep="")
        else:
            nodes_idx.tofile(self._file, sep="")

        self._file.write(b'\n')
        self._file.write(b'CELL_TYPES ')
        self._file.write(f'{len(cells)}'.encode('ascii'))
        self._file.write(b'\n')
        if size == 4:
            # Quad
            cell_type = 9 * np.ones(len(cells), dtype=np.int32)
        elif size == 8:
            # Hexahedron
            cell_type = 12 * np.ones(len(cells), dtype=np.int32)
        else:
            return
        if np.little_endian:
            cell_type.byteswap().tofile(self._file, sep="")
        else:
            cell_type.tofile(self._file, sep="")
        self._file.write(b'\n')

    def __write_data(self):
        self._file.write(b'CELL_DATA ')
        self._file.write(f'{len(self.grid.cells)}'.encode('ascii'))
        self._file.write(b'\n')

        self._file.write(b'SCALARS density float 1\nLOOKUP_TABLE default\n')
        rho = np.array(self.lattice.rho, dtype=np.float32)
        if np.little_endian:
            rho.byteswap().tofile(self._file, sep="")
        else:
            rho.byteswap().tofile(self._file, sep="")
        self._file.write(b'\n')

        self._file.write(b'VECTORS velocity float\n')
        u = np.array(self.lattice.u, dtype=np.float32)
        if u.shape[1] == 2:
            u = np.concatenate((u, np.zeros((u.shape[0], 1), dtype=np.float32)), axis=1)
        u = u.flatten()
        if np.little_endian:
            u.byteswap().tofile(self._file, sep="")
        else:
            u.byteswap().tofile(self._file, sep="")
        self._file.write(b'\n')

    def __write_cell_info(self):
        pass
        #self._file.write(b'SCALARS cell_type int 1\nLOOKUP_TABLE default\n')
        #cell_info = np.array([0 if cell.bc is None else cell.bc.boundary_condition
        #                      for cell in self.grid.cells], dtype=np.int32)
        #if np.little_endian:
        #    cell_info.byteswap().tofile(self._file, sep="")
        #else:
        #    cell_info.byteswap().tofile(self._file, sep="")
        #self._file.write(b'\n')

    def __write_header_ascii(self):
        self._file.write(f'# vtk DataFile Version 3.0\n')
        self._file.write(f'lbm solver data\n')
        self._file.write(f'ASCII\n')
        self._file.write(f'DATASET UNSTRUCTURED_GRID\n')

    def __write_points_ascii(self):
        points = self.grid.nodes
        self._file.write(f'POINTS {len(points)} float\n')
        for node in points:
            self._file.write(f'{node.x} {node.y} {node.z}\n')

    def __write_cells_ascii(self):
        cells = self.grid.cells
        #size = np.int32(2**self.lattice.stencil.d)
        size = len(cells[0].nodes)
        self._file.write(f'CELLS {len(cells)} {int((size + 1) * len(cells))}\n')
        if 'connectivity' in self._cache:
            nodes_idx = self._cache['connectivity']
        else:
            nodes = np.array([node for node in [cell.nodes for cell in cells]])
            nodes_idx = np.array([node.index for node in nodes.flatten()], dtype=np.int32)
            nodes_idx = np.insert(nodes_idx, np.arange(size, nodes_idx.size, size), size)
            nodes_idx = np.insert(nodes_idx, 0, size)
            nodes_idx = np.reshape(nodes_idx, (-1, size + 1))
            self._cache['connectivity'] = nodes_idx
        for n_idx in nodes_idx:
            self._file.write(f'{" ".join(map(str, n_idx))}\n')
        if size == 4:
            # Quad
            cell_type = 9
        elif size == 8:
            # Hexahedron
            cell_type = 12
        else:
            return
        self._file.write(f'CELL_TYPES {len(cells)}\n')
        for _ in cells:
            self._file.write(f'{cell_type}\n')

    def __write_data_ascii(self):
        self._file.write(f'CELL_DATA ')
        self._file.write(f'{len(self.grid.cells)}\n')

        self._file.write(f'SCALARS density float 1\nLOOKUP_TABLE default\n')
        rho = np.array(self.lattice.rho, dtype=np.float32)
        for r in rho:
            self._file.write(f'{r}\n')

        self._file.write(f'VECTORS velocity float\n')
        u = np.array(self.lattice.u, dtype=np.float32)
        if u.shape[1] == 2:
            u = np.concatenate((u, np.zeros((u.shape[0], 1), dtype=np.float32)), axis=1)
        for ui in u:
            self._file.write(f'{" ".join(map(str, ui))}\n')

