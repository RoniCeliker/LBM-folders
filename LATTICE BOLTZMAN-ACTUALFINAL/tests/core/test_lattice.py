import pytest
from pytest import approx
import numpy as np
from lbm.core.lattice import Lattice
from lbm.mesh.mesh_generator import MeshGenerator
from lbm.core.stencil import Stencil


@pytest.fixture
def mesh_box_2d():
    n = 6
    mesh_gen = MeshGenerator()
    boundary_names = {"west": "WALL", "east": "WALL", "south": "WALL", "north": "WALL"}
    mesh_gen.create_block(nx=n, ny=n, boundary_names=boundary_names)
    mesh = mesh_gen.mesh
    mesh.setup()
    return mesh


@pytest.fixture
def mesh_box_3d():
    n = 6
    mesh_gen = MeshGenerator()
    boundary_names = {"west": "WALL", "east": "WALL",
                      "south": "WALL", "north": "WALL",
                      "back": "WALL", "front": "WALL"}
    mesh_gen.create_block(nx=n, ny=n, nz=n, boundary_names=boundary_names)
    mesh = mesh_gen.mesh
    mesh.setup()
    return mesh


def lattice_box(d: int, q: int, mesh):
    stencil = Stencil(d, q)
    lattice = Lattice(stencil=stencil, mesh=mesh)
    # Initiate a boundary condition for the edges
    bc = {"boundary_conditions": [{"boundary_name": "WALL", "boundary_type": "ORPHAN"}]}
    mesh.setup_bc(bc)
    lattice.setup_streaming()
    # Set unique values for every population in f
    n = len(lattice.mesh.cells)
    f = np.zeros_like(lattice.f)
    np.put(f, np.arange(int(n * stencil.q)), np.arange(0, n * stencil.q))
    lattice.f[:] = f[:]
    lattice.f0[:] = f[:]
    # Set unique velocities
    np.put(lattice.u[:, 0], np.arange(n, dtype=np.int32), np.arange(1, n + 1, dtype=np.float64))
    if d > 1:
        np.put(lattice.u[:, 1], np.arange(n, dtype=np.int32), np.arange(n + 1, -1, -1, dtype=np.float64))
        lattice.u[:, 1] *= -1
    if d > 2:
        lattice.u[:, 2] = 0.5*(lattice.u[:, 0] + lattice.u[:, 1])
    return lattice


@pytest.mark.parametrize("d,q", [(2, 9)])
def test_streaming_2d(mesh_box_2d, d, q):
    lattice = lattice_box(d=d, q=q, mesh=mesh_box_2d)
    stencil = lattice.stencil
    mesh = lattice.mesh
    lattice.streaming()

    cell = mesh.find_cell([2.5, 2.5])
    for c in stencil.c:
        i = stencil.c_pop(c)
        cell_n = cell.find_neighbour(-c).index
        assert lattice.f0[i, cell_n] == approx(lattice.f[i, cell.index])


@pytest.mark.parametrize("d,q", [(3, 15), (3, 19), (3, 27)])
def test_streaming_3d(mesh_box_3d, d, q):
    lattice = lattice_box(d=d, q=q, mesh=mesh_box_3d)
    stencil = lattice.stencil
    mesh = lattice.mesh
    lattice.streaming()

    cell = mesh.find_cell([2.5, 2.5, 2.5])
    for c in stencil.c:
        i = stencil.c_pop(c)
        cell_n = cell.find_neighbour(-c).index
        assert lattice.f0[i, cell_n] == approx(lattice.f[i, cell.index])


def test_equilibrium_2d(mesh_box_2d):
    lattice = lattice_box(d=2, q=9, mesh=mesh_box_2d)
    stencil = lattice.stencil
    mesh = lattice.mesh

    # a + (b - a). * rand(100, 1);
    a = 1.0
    b = 0.0
    for i in range(lattice.f.shape[0]):
        for j in range(lattice.f.shape[1]):
            lattice.f[i, j] = a + (b - a) * np.random.rand()
    a = 0.1
    b = -0.1
    for i in range(lattice.u.shape[0]):
        for j in range(lattice.u.shape[1]):
            lattice.u[i, j] = a + (b - a) * np.random.rand()
    lattice.equilibrium()
    cell = mesh.find_cell([2.5, 2.5])
    print(cell)
    idx = cell.index
    print(lattice.feq[:, idx])
    print(lattice.feq.shape)
    feq = lattice.feq
    f = lattice.f
    i_opp = stencil.i_opp
    f_plus = 0.5*(f[:, idx] + f[i_opp, idx])
    f_minus = 0.5 * (f[:, idx] - f[i_opp, idx])
    f_eq_plus = 0.5*(feq[:, idx] + feq[i_opp, idx])
    f_eq_minus = 0.5 * (feq[:, idx] - feq[i_opp, idx])
    print(f"i_opp = {i_opp}")
    print(f"f     = {f[:, idx]}")
    print(f"f_opp = {f[i_opp, idx]}")
    print(f"f+    = {f_plus}")
    print(f"f-    = {f_minus}")
    print(f"feq+  = {f_eq_plus}")
    print(f"feq-  = {f_eq_minus}")
    print()
    print(f"f+ - feq+  = {f_plus - f_eq_plus}")
    print(f"f- - feq-  = {f_minus - f_eq_minus}")
    print()
    print(f"f - feq {f[:, idx] - feq[:, idx]}")
    print(f"f - feq opp {f[i_opp, idx] - feq[i_opp, idx]}")
    assert False
