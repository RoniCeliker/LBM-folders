import numpy as np
from lbm.core.bc_factory import bc_factory
from lbm.core.bc.boundary_condition import BoundaryCondition


class Boundary:
    def __init__(self, name: str, faces=None, index: int = 0):
        self.index = index
        self.name = name
        if faces is None:
            self.faces = []
        else:
            self.faces = faces
        self._bc = None

    def __str__(self):
        return f"<Boundary:{self._index}_({self.name})_{len(self.faces)}>"

    def __repr__(self):
        return f"<Boundary:{self._index}_({self.name})_{len(self.faces)}>"

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index):
        self._index = np.uint32(index)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def faces(self):
        return self._faces

    @faces.setter
    def faces(self, faces):
        self._faces = faces
        
    @property
    def bc(self) -> BoundaryCondition:
        return self._bc

    @bc.setter
    def bc(self, bc: BoundaryCondition):
        self._bc = bc

    def set_bc(self, data):
        if 'data' in data:
            self._bc = bc_factory.create(data['boundary_type'], **data['data'])
        else:
            self._bc = bc_factory.create(data['boundary_type'])

    def ensure_faces(self):
        for face in self.faces:
            face.boundary = self

    def renumber_faces(self):
        for i in range(len(self.faces)):
            self.faces[i].index = i

    def delete_face(self, face):
        if face in self.faces:
            self.faces.remove(face)

    def find_face(self, r):
        r = np.array(r, np.float64)
        r0 = np.array([face.coordinate for face in self.faces])
        idx = np.where(np.linalg.norm(r0 - r, axis=1) < 1e-3)[0]
        if idx.size > 0:
            return self.faces[idx[0]]
        else:
            return None
