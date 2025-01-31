import numpy as np
from numba import cuda


class Stencil:
    """
    The Stencil class should be considered constant for a given d, q
    Speed of sound "cs" is only stored here for conveniance, also constant
    """
    def __init__(self, d: int, q: int):
        self._d = np.int32(d)                                    # Number of dimensions
        self._q = np.int32(q)                                    # Number of characteristics
        self._w = np.zeros(self.q, dtype=np.float32)             # Weights
        self._c = np.zeros((self.d, self.q), dtype=np.float32)   # Characteristic velocity
        #array sent to gpu
        if self.d == 2 and self.q == 9:
            self.__d2q9()
        elif self.d == 3 and self.q == 15:
            self.__d3q15()
        elif self.d == 3 and self.q == 19:
            self.__d3q19()
        elif self.d == 3 and self.q == 27:
            self.__d3q27()
        else:
            raise Exception(f'Invalid choice of DdQq: "D{self.d}Q{self.q}"')
        self._i_opp = self.__opposite()
        self.cs = np.float32(1.0 / np.sqrt(3))
        self.cs_2 = np.float32(self.cs**2)
        self.cs_4 = np.float32(self.cs**4)
        self.inv_cs = np.float32(1.0/self.cs)
        self.inv_cs_2 = np.float32(1.0/self.cs_2)
        self.inv_cs_4 = np.float32(1.0/self.cs_4)

        self._i_sym = np.zeros((self.q, self.q), dtype=np.int32)
        for i in range(self.q):
            for j in range(self.q):
                c_ij = self.__symmetry_normal(self.c[i], self.c[j])
                ij = self.c_pop(c_ij)
                self._i_sym[i, j] = ij
    
    def __str__(self):  # pragma: no cover
        return f"Stencil<D{self.d},Q{self.q}>"

    @property
    def q(self):
        return self._q

    @property
    def d(self):
        return self._d

    @property
    def w(self):
        return self._w

    @property
    def c(self):
        return self._c

    @property
    def i_opp(self):
        return self._i_opp

    @property
    def i_sym(self):
        return self._i_sym

    def c_pop(self, vec, theta_tol=2.0) -> int:
        vec_norm = np.linalg.norm(vec)
        c_norm = np.linalg.norm(self.c, axis=1)
        if vec_norm < 1e-9:
            index = np.argmin(c_norm)
        else:
            c_norm[c_norm == 0] = np.inf
            dot = np.dot(self.c, vec) / (vec_norm * c_norm)
            # If the input vector does not lie within theta_tol, return q
            if np.any(dot > np.cos(np.deg2rad(theta_tol))):
                index = np.argmax(dot)
            else:
                index = self.q
        return index

    def __opposite(self):
        opposite = np.zeros(self.q, dtype=np.int32)
        for i in range(self.q):
            opposite[i] = self.c_pop(-self.c[i])
        return opposite

    @staticmethod
    def __symmetry_normal(normal, c):
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            return np.zeros(1, dtype=np.float32)
        normal = normal/norm
        return c - 2*np.dot(c, normal)*normal

    def __d2q9(self):
        self._c = np.array([[0, 0],
                            [1, 0], [0, 1], [-1, 0], [0, -1],
                            [1, 1], [-1, 1], [-1, -1], [1, -1]],
                           dtype=np.float32)
        self._w = np.array([4 / 9,
                            1 / 9, 1 / 9, 1 / 9, 1 / 9,
                            1 / 36, 1 / 36, 1 / 36, 1 / 36],
                           dtype=np.float32)

    def __d3q15(self):
        self._c = np.array([[0, 0, 0],
                            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
                            [1, 1, 1], [-1, -1, -1], [1, 1, -1], [-1, -1, 1],
                            [1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1]],
                           dtype=np.float32)
        self._w = np.array([2 / 9,
                            1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9,
                            1 / 72, 1 / 72, 1 / 72, 1 / 72, 1 / 72, 1 / 72, 1 / 72, 1 / 72],
                           dtype=np.float32)

    def __d3q19(self):
        self._c = np.array([[0, 0, 0],
                            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
                            [1, 1, 0], [-1, -1, 0], [1, 0, 1], [-1, 0, -1], [0, 1, 1], [0, -1, -1],
                            [1, -1, 0], [-1, 1, 0], [1, 0, -1], [-1, 0, 1], [0, 1, -1], [0, -1, 1]],
                           dtype=np.float32)
        self._w = np.array([1 / 3,
                            1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18, 1 / 18,
                            1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36,
                            1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
                           dtype=np.float32)

    def __d3q27(self):
        self._c = np.array([[0, 0, 0],
                            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
                            [1, 1, 0], [-1, -1, 0], [1, 0, 1], [-1, 0, -1], [0, 1, 1], [0, -1, -1],
                            [1, -1, 0], [-1, 1, 0], [1, 0, -1], [-1, 0, 1], [0, 1, -1], [0, -1, 1],
                            [1, 1, 1], [-1, -1, -1], [1, 1, -1], [-1, -1, 1],
                            [1, -1, 1], [-1, 1, -1], [-1, 1, 1], [1, -1, -1]],
                           dtype=np.float32)
        self._w = np.array([8 / 27,
                            2 / 27, 2 / 27, 2 / 27, 2 / 27, 2 / 27, 2 / 27,
                            1 / 54, 1 / 54, 1 / 54, 1 / 54, 1 / 54, 1 / 54,
                            1 / 54, 1 / 54, 1 / 54, 1 / 54, 1 / 54, 1 / 54,
                            1 / 216, 1 / 216, 1 / 216, 1 / 216,
                            1 / 216, 1 / 216, 1 / 216, 1 / 216],
                           dtype=np.float32)
