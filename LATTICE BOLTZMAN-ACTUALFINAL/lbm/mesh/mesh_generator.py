import numpy as np
from lbm.mesh.mesh import Mesh
from lbm.mesh.boundary import Boundary
from lbm.mesh.node import Node
from lbm.mesh.face import Face
from lbm.mesh.cell import Cell


class MeshGenerator:
    def __init__(self, mesh: Mesh = None):
        if mesh is None:
            self.mesh = Mesh()
        else:
            self.mesh = mesh
        self._boundary_nodes = []
        self._r = None
        self._idx = None

    def create_block(self, nx: int = 1, ny: int = None, nz: int = None,
                     x_offset: int = 0, y_offset: int = 0, z_offset: int = 0,
                     boundary_names: dict = None, renumber=True, replace_nodes=True):
        if boundary_names is None:
            boundary_names = {"west": "WALL", "east": "WALL",
                              "south": "WALL", "north": "WALL",
                              "back": "WALL", "front": "WALL"}
        if nx is None:
            raise Exception("Can not create a block")
        elif ny is None:
            n = np.array([nx])
        elif nz is None:
            n = np.array([nx, ny])
        else:
            n = np.array([nx, ny, nz])

        nodes = self._create_nodes(n, np.array([x_offset, y_offset, z_offset]))
        if replace_nodes:
            self._replace_nodes(n, nodes)
        cells = self._create_cells(n, nodes)
        faces = self._create_faces(n, nodes)
        self._append_to_mesh(nodes, cells, faces, boundary_names)
        if renumber:
            self.renumber_all()

    def renumber_all(self):
        self.mesh.renumber_nodes()
        self.mesh.renumber_cells()
        for key in self.mesh.boundaries:
            self.mesh.boundaries[key].renumber_faces()
            self.mesh.boundaries[key].ensure_faces()

    def _append_to_mesh(self, nodes, cells: list[Cell], faces: dict, boundary_names):
        if self.mesh.nodes:
            nodes = nodes.flatten().tolist()
        else:
            nodes = [node for node in nodes.flatten() if node not in self.mesh.nodes]
        self.mesh.nodes += nodes
        self.mesh.cells += cells
        for key in boundary_names:
            if not faces[key] or boundary_names[key] is None:
                continue
            if boundary_names[key] in [b for b in self.mesh.boundaries]:
                self.mesh.boundaries[boundary_names[key]].faces += faces[key]
            else:
                boundary = Boundary(name=boundary_names[key], faces=faces[key])
                self.mesh.boundaries[boundary_names[key]] = boundary

    def _create_nodes(self, n: np.ndarray, n_offset: np.ndarray):
        if n.size == 1:
            x = np.arange(0, n[0] + 1, dtype=np.float64) + n_offset[0]
            r = x
        elif n.size == 2:
            x = np.arange(0, n[0] + 1, dtype=np.float64) + n_offset[0]
            y = np.arange(0, n[1] + 1, dtype=np.float64) + n_offset[1]
            xx, yy = np.meshgrid(x, y, indexing='ij')
            r = np.array([xx.flatten(), yy.flatten()]).T
        elif n.size == 3:
            x = np.arange(0, n[0] + 1, dtype=np.float64) + n_offset[0]
            y = np.arange(0, n[1] + 1, dtype=np.float64) + n_offset[1]
            z = np.arange(0, n[2] + 1, dtype=np.float64) + n_offset[2]
            xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
            r = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
        else:
            raise Exception
        nodes = [Node(r=r[i], index=i) for i in range(r.shape[0])]
        nodes = np.reshape(nodes, n + 1)
        self._cache_boundary_nodes(nodes)
        return nodes

    def _cache_boundary_nodes(self, nodes):
        if len(nodes.shape) == 2:
            self._boundary_nodes += nodes[:, [0, -1]].flatten().tolist()
            self._boundary_nodes += nodes[[0, -1], :].flatten().tolist()
        elif len(nodes.shape) == 3:
            self._boundary_nodes += nodes[:, :, [0, -1]].flatten().tolist()
            self._boundary_nodes += nodes[:, [0, -1], :].flatten().tolist()
            self._boundary_nodes += nodes[[0, -1], :, :].flatten().tolist()

    def _find_boundary_node(self, r):
        r = np.array(r, np.float64)
        idx_left0 = 0
        idx_right0 = -1
        idx = 0
        i = -1
        for ri in r[::-1]:
            idx_left = np.searchsorted(self._r[idx_left0:idx_right0, i], ri, side='left')
            idx_right = np.searchsorted(self._r[idx_left0:idx_right0, i], ri, side='right')
            idx += idx_left
            idx_right0 = idx_left0 + idx_right
            idx_left0 += idx_left
            i -= 1
        node = self._boundary_nodes[self._idx[idx]]
        if np.linalg.norm(node.r - r) < 1e-3:
            return node
        else:
            return None

    def _replace_nodes(self, n: np.ndarray, nodes: np.ndarray):
        self._r = np.array([node.r for node in self._boundary_nodes])
        self._idx = np.lexsort([self._r[:, i] for i in range(self._r.shape[1])])
        self._r = self._r[self._idx]
        if self.mesh.nodes:
            if n.size == 2:
                for i in range(n[0] + 1):
                    f_node = self._find_boundary_node(nodes[i, 0].r)
                    if f_node is not None:
                        nodes[i, 0] = f_node
                    f_node = self._find_boundary_node(nodes[i, -1].r)
                    if f_node is not None:
                        nodes[i, -1] = f_node

                for i in range(n[1] + 1):
                    f_node = self._find_boundary_node(nodes[0, i].r)
                    if f_node is not None:
                        nodes[0, i] = f_node
                    f_node = self._find_boundary_node(nodes[-1, i].r)
                    if f_node is not None:
                        nodes[-1, i] = f_node
            elif n.size == 3:
                for i in range(n[0] + 1):
                    for j in range(n[1] + 1):
                        f_node = self._find_boundary_node(nodes[i, j, 0].r)
                        if f_node is not None:
                            nodes[i, j, 0] = f_node
                        f_node = self._find_boundary_node(nodes[i, j, -1].r)
                        if f_node is not None:
                            nodes[i, j, -1] = f_node

                for i in range(n[0] + 1):
                    for j in range(n[2] + 1):
                        f_node = self._find_boundary_node(nodes[i, 0, j].r)
                        if f_node is not None:
                            nodes[i, 0, j] = f_node
                        f_node = self._find_boundary_node(nodes[i, -1, j].r)
                        if f_node is not None:
                            nodes[i, -1, j] = f_node

                for i in range(n[1] + 1):
                    for j in range(n[2] + 1):
                        f_node = self._find_boundary_node(nodes[0, i, j].r)
                        if f_node is not None:
                            nodes[0, i, j] = f_node
                        f_node = self._find_boundary_node(nodes[-1, i, j].r)
                        if f_node is not None:
                            nodes[-1, i, j] = f_node

    @staticmethod
    def _create_cells(n: np.ndarray, nodes: np.ndarray) -> list[Cell]:
        cells = []
        counter = 0
        if n.size == 2:
            for xi in range(n[0]):
                for yi in range(n[1]):
                    tmp = [nodes[xi, yi],
                           nodes[xi + 1, yi],
                           nodes[xi + 1, yi + 1],
                           nodes[xi, yi + 1]]
                    cells.append(Cell(nodes=tmp, index=counter))
                    counter += 1
        elif n.size == 3:
            for xi in range(n[0]):
                for yi in range(n[1]):
                    for zi in range(n[2]):
                        tmp = [nodes[xi, yi, zi],
                               nodes[xi + 1, yi, zi],
                               nodes[xi + 1, yi + 1, zi],
                               nodes[xi, yi + 1, zi],
                               nodes[xi, yi, zi + 1],
                               nodes[xi + 1, yi, zi + 1],
                               nodes[xi + 1, yi + 1, zi + 1],
                               nodes[xi, yi + 1, zi + 1]]
                        cells.append(Cell(nodes=tmp, index=counter))
                        counter += 1
        return cells

    @staticmethod
    def _create_faces(n: np.ndarray, nodes: np.ndarray) -> dict:
        faces = {"west": [], "east": [], "south": [], "north": [], "back": [], "front": []}
        if n.size == 2:
            for i in range(n[0]):
                faces["south"].append(Face(nodes=[nodes[i, 0], nodes[i + 1, 0]]))
                faces["north"].append(Face(nodes=[nodes[i + 1, n[1]], nodes[i, n[1]]]))

            for i in range(n[1]):
                faces["west"].append(Face(nodes=[nodes[0, i + 1], nodes[0, i]]))
                faces["east"].append(Face(nodes=[nodes[n[0], i], nodes[n[0], i + 1]]))
        elif n.size == 3:
            for i in range(n[0]):
                for j in range(n[1]):
                    faces["back"].append(Face(nodes=[nodes[i, j, 0], nodes[i, j + 1, 0],
                                                     nodes[i + 1, j + 1, 0], nodes[i + 1, j, 0]]))
                    faces["front"].append(Face(nodes=[nodes[i, j, n[2]], nodes[i + 1, j, n[2]],
                                                      nodes[i + 1, j + 1, n[2]], nodes[i, j + 1, n[2]]]))

            for i in range(n[0]):
                for j in range(n[2]):
                    faces["south"].append(Face(nodes=[nodes[i, 0, j], nodes[i + 1, 0, j],
                                                      nodes[i + 1, 0, j + 1], nodes[i, 0, j + 1]]))
                    faces["north"].append(Face(nodes=[nodes[i, n[1], j], nodes[i, n[1], j + 1],
                                                      nodes[i + 1, n[1], j + 1], nodes[i + 1, n[1], j]]))

            for i in range(n[1]):
                for j in range(n[2]):
                    faces["west"].append(Face(nodes=[nodes[0, i, j], nodes[0, i, j + 1],
                                                     nodes[0, i + 1, j + 1], nodes[0, i + 1, j]]))
                    faces["east"].append(Face(nodes=[nodes[n[0], i, j], nodes[n[0], i + 1, j],
                                                     nodes[n[0], i + 1, j + 1], nodes[n[0], i, j + 1]]))
        return faces


def main():
    mesh_gen = MeshGenerator()

    #n = 6
    #boundary_names = {"west": "WEST", "east": "EAST",
    #                  "south": "SOUTH", "north": "NORTH",
    #                  "back": "BACK", "front": "FRONT"}
    #mesh_gen.create_block(nx=n, ny=n, nz=n, boundary_names=boundary_names)

    n = 3
    boundary_names = {"west": "WALL", "east": "WALL",
                      "south": "WALL", "north": "WALL"}
    mesh_gen.create_block(nx=n, ny=n, boundary_names=boundary_names)
    mesh_gen.create_block(nx=n, ny=n, y_offset=n, boundary_names=boundary_names)

    mesh = mesh_gen.mesh
    mesh.setup()
    print("====================================")
    cell = mesh.find_cell([1.5, 3.5])
    print(cell, cell.coordinate)
    print(cell, cell.neighbours)
    cell = mesh.find_cell([1.5, 2.5])
    print(cell, cell.coordinate)
    print(cell, cell.neighbours)
    mesh.plot_mesh_2d()
    #export = Export(grid=mesh)
    #export.write_vtk("test.vtk", write_cell_info=True)
    #mesh.write_ffa("test.bmsh")


def main2():
    n = 4
    boundary_names = {"west": "WEST", "east": "EAST",
                      "south": "SOUTH", "north": "NORTH",
                      "back": "BACK", "front": "FRONT"}
    mesh_gen = MeshGenerator()
    mesh_gen.create_block(nx=2*n, ny=n, nz=n, boundary_names=boundary_names)

    mesh = mesh_gen.mesh
    mesh.setup()
    #cell = mesh.cells[0]
    #print(cell, cell.neighbours)
    #print(mesh.boundaries["WALL"].faces)
    mesh.write_ffa_connectivity("tmp.bcon")
    mesh.read_ffa_connectivity("tmp.bcon")


if __name__ == "__main__":
    main2()
