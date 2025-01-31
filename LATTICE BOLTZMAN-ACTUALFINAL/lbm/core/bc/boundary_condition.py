from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from lbm.core.bc import BoundaryType, BoundaryClass
from lbm.core.stencil import Stencil


def bc_priority(bc_list: list[BoundaryCondition]) -> BoundaryCondition:
    bc_list = np.array(bc_list)
    boundary_type = np.array([bc.boundary_type for bc in bc_list])
    idx = np.where(boundary_type == np.min(boundary_type))[0]
    boundary_class = np.array([bc.boundary_class for bc in bc_list[idx]])
    return bc_list[idx[np.argmin(boundary_class)]]


class BoundaryCondition(ABC):
    def __init__(self, name: str, boundary_type: BoundaryType, boundary_class: BoundaryClass):
        self.name = name
        self.boundary_type = boundary_type
        self.boundary_class = boundary_class
        self._streaming_map = np.empty((0, 0), dtype=np.uint32)
        self._capacity = np.uint32(0)
        self._size = np.uint32(0)
        self.wall_data = np.float64(0)

    def __repr__(self):
        return f"<{self.boundary_type}:TYPE:{self.boundary_class}:{self.name}>"

    @property
    def streaming_map(self) -> np.ndarray:
        return self._streaming_map

    @streaming_map.setter
    def streaming_map(self, streaming_map):
        self._streaming_map = streaming_map
        self._size = np.uint32(self._streaming_map.shape[0])
        if self._size > self._capacity:
            self._capacity = self._size

    @property
    def size(self) -> np.uint32:
        return self._size

    @property
    def capacity(self) -> np.uint32:
        return self._capacity

    def push_back(self, streaming_map_i):
        if self._size >= self._capacity:
            self._capacity = np.uint32(np.max([np.uint32(2) * self._capacity, np.uint32(1)]))
            self._streaming_map.resize((self._capacity, self._streaming_map.shape[1]))
        self._streaming_map[self._size] = streaming_map_i
        self._size += np.uint32(1)

    def shrink_to_fit(self):
        self._streaming_map.resize((self._size, self._streaming_map.shape[1]))
        self._capacity = self._size

    def reserve(self, new_capacity: np.uint32):
        new_capacity = np.uint32(new_capacity)
        if new_capacity > self._capacity:
            self._streaming_map.resize((new_capacity, self._streaming_map.shape[1]))
            self._capacity = new_capacity

    @abstractmethod
    def streaming(self, lattice, stencil):  # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def setup_streaming(self, pop_idx, cell, faces, stencil: Stencil):  # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def setup_boundary_data(self, stencil: Stencil):  # pragma: no cover
        raise NotImplementedError
