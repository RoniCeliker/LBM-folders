from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod


class Element(ABC):
    def __init__(self, nodes=None, index: int = 0):
        self.index = index
        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes
        self.neighbours = []

    def __str__(self):
        return f"<Element:{self._index}>"

    def __repr__(self):
        return f"<Element:{self._index}>"

    @property
    def nodes(self) -> list:
        return self._nodes

    @nodes.setter
    def nodes(self, nodes):
        self._nodes = nodes

    @property
    def neighbours(self):
        return self._neighbours

    @neighbours.setter
    def neighbours(self, neighbours: list[Element]):
        self._neighbours = neighbours

    @property
    def index(self) -> np.uint32:
        return self._index

    @index.setter
    def index(self, index):
        self._index = np.uint32(index)

    @property
    def coordinate(self) -> np.ndarray:
        return np.mean(np.array([node.r for node in self.nodes], dtype=np.float64), axis=0)

    @abstractmethod
    def create_connectivity(self) -> None:  # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def find_neighbour(self, traverse):  # pragma: no cover
        raise NotImplementedError

    def delete(self) -> None:
        for node in self.nodes:
            node.remove_element(self)
        for neighbour in self.neighbours:
            if neighbour is not None:
                neighbour.neighbours[neighbour.neighbours.index(self)] = None
        self.neighbours = [None for _ in self.neighbours]
