import numpy as np
from lbm.mesh.element import Element


class Face(Element):
    def __init__(self, nodes=None, index: int = 0, boundary=None):
        super().__init__(nodes=nodes, index=index)
        self.boundary = boundary

    def __str__(self):
        return f"<Face:{self.index}>"

    def __repr__(self):
        return f"<Face:{self.index}>"

    @property
    def boundary(self):
        return self._boundary

    @boundary.setter
    def boundary(self, boundary):
        self._boundary = boundary

    @property
    def normal(self) -> np.ndarray:
        if len(self.nodes) == 2:
            r = self.nodes[0].r - self.nodes[1].r
            n = np.array([-r[1], r[0]], dtype=np.float64)
            return n/np.linalg.norm(n)
        elif len(self.nodes) == 4:
            n = np.cross(self.nodes[2].r - self.nodes[0].r, self.nodes[3].r - self.nodes[1].r)
            return n/np.linalg.norm(n)

    def flip_normal(self) -> None:
        self.nodes.reverse()

    def create_connectivity(self) -> None:
        # TODO if no neighbouring cell is found, swap normal and look again
        cells = [element for element in self.nodes[0].elements if not isinstance(element, Face)]

        if len(self.nodes) > 2:  # 3D
            cells += [element for element in self.nodes[2].elements
                      if not isinstance(element, Face) and element not in cells]

            idx = cells[0].get_cell_face()
            # A 3D cell has six faces, arrange those faces in a nested list of nodes
            nodes = [[[cell.nodes[i] for i in j] for j in idx] for cell in cells]
            i = 0
            for cell_node in nodes:
                for node in cell_node:
                    # The cell face node order is reversed compare to self.nodes
                    if node == self.nodes[::-1] or \
                       node[-1:] + node[0:-1] == self.nodes[::-1] or \
                       node[-2:] + node[0:-2] == self.nodes[::-1] or \
                       node[-3:] + node[0:-3] == self.nodes[::-1]:
                        self.neighbours = [cells[i]]
                        cells[i].set_cardinal_neighbour(self.normal, self)
                        """ print(self.neighbours) """
                        break
                i += 1
        if len(self.nodes) > 1:  # 2D
            nodes = [cell.nodes for cell in cells]
            i = 0
            for node in nodes:
                # Looking for an ordered sublist (self.nodes) in the list of nodes (node)
                if all(n in j for j in [iter(node)] for n in self.nodes) or \
                   all(n in j for j in [iter(node[-2:] + node[0:-2])] for n in self.nodes):
                    self.neighbours = [cells[i]]
                    cells[i].set_cardinal_neighbour(self.normal, self)
                    break
                i += 1

    def find_neighbour(self, traverse):
        pass
