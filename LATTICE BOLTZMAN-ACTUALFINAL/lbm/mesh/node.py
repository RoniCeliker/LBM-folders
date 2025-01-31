import numpy as np


class Node:
    def __init__(self, r, index: int = 0):
        self.index = index
        self.r = r
        self.elements = []

    def __str__(self):
        return f"<Node:{self._index}_{self._r}>"

    def __repr__(self):
        return f"<Node:{self._index}_{self._r}>"

    @property
    def index(self) -> np.uint32:
        return self._index

    @index.setter
    def index(self, index):
        self._index = np.uint32(index)

    @property
    def r(self) -> np.ndarray:
        return self._r

    @r.setter
    def r(self, r):
        self._r = np.array(r, dtype=np.float64)

    @property
    def x(self):
        return self._r[0]

    @property
    def y(self):
        return self._r[1]

    @property
    def z(self):
        return self._r[2]

    @property
    def elements(self) -> list:
        return self._elements

    @elements.setter
    def elements(self, elements):
        self._elements = elements

    def append_element(self, element) -> None:
        if element not in self.elements:
            self.elements.append(element)

    def remove_element(self, element) -> None:
        if element in self.elements:
            self.elements.remove(element)

    def clear(self) -> None:
        self.elements = None
