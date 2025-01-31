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
    boundary_names = {"west": "WEST", "east": "EAST", "south": "SOUTH", "north": "NORTH"}
    mesh_gen.create_block(nx=n, ny=n, boundary_names=boundary_names)
    mesh = mesh_gen.mesh
    mesh.setup()
    return mesh


@pytest.fixture
def mesh_box_2d_corner_bc():
    n = 3
    mesh_gen = MeshGenerator()

    # Southwest
    boundary_names = {"west": "SOUTHWEST", "east": None, "south": "SOUTHWEST", "north": None}
    mesh_gen.create_block(nx=n, ny=n, x_offset=0, y_offset=0, boundary_names=boundary_names)
    # Northwest
    boundary_names = {"west": "NORTHWEST", "east": None, "south": None, "north": "NORTHWEST"}
    mesh_gen.create_block(nx=n, ny=n, x_offset=0, y_offset=n, boundary_names=boundary_names)
    # Northeast
    boundary_names = {"west": None, "east": "NORTHEAST", "south": None, "north": "NORTHEAST"}
    mesh_gen.create_block(nx=n, ny=n, x_offset=n, y_offset=n, boundary_names=boundary_names)
    # Southeast
    boundary_names = {"west": None, "east": "SOUTHEAST", "south": "SOUTHEAST", "north": None}
    mesh_gen.create_block(nx=n, ny=n, x_offset=n, y_offset=0, boundary_names=boundary_names)

    mesh = mesh_gen.mesh
    mesh.setup()
    return mesh


@pytest.fixture
def mesh_box_2d_convex_corner():
    n = 3
    mesh_gen = MeshGenerator()

    # Northeast
    boundary_names = {"west": None, "east": "WALL", "south": None, "north": "WALL"}
    mesh_gen.create_block(nx=n, ny=n, x_offset=0, y_offset=0, boundary_names=boundary_names)
    # West
    boundary_names = {"west": "WALL", "east": None, "south": "WEST", "north": "WEST"}
    mesh_gen.create_block(nx=n, ny=n, x_offset=-n, y_offset=0, boundary_names=boundary_names)
    # South
    boundary_names = {"west": "SOUTH", "east": "SOUTH", "south": "WALL", "north": None}
    mesh_gen.create_block(nx=n, ny=n, x_offset=0, y_offset=-n, boundary_names=boundary_names)

    mesh = mesh_gen.mesh
    mesh.setup()
    return mesh


@pytest.fixture
def mesh_box_3d():
    n = 6
    mesh_gen = MeshGenerator()
    boundary_names = {"west": "WEST", "east": "EAST",
                      "south": "SOUTH", "north": "NORTH",
                      "back": "BACK", "front": "FRONT"}
    mesh_gen.create_block(nx=n, ny=n, nz=n, boundary_names=boundary_names)
    mesh = mesh_gen.mesh
    mesh.setup()
    return mesh


def lattice_box(d: int, q: int, mesh):
    stencil = Stencil(d=d, q=q)
    lattice = Lattice(stencil=stencil, mesh=mesh)
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


def anti_bounce_back_wall_data(cell, cell_n, density, lattice, stencil):
    u_wall = 1.5 * lattice.u[cell.index] - 0.5 * lattice.u[cell_n.index]
    return 2.0 * stencil.w * density\
           * (1 + 0.5 * np.dot(stencil.c, u_wall) ** 2 / stencil.cs ** 4
              - 0.5 * np.dot(u_wall, u_wall) / stencil.cs ** 2)


def test_bounce_back_wall_d2q9(mesh_box_2d):
    lattice = lattice_box(d=2, q=9, mesh=mesh_box_2d)
    stencil = lattice.stencil
    mesh = lattice.mesh
    bc_name = "BOUNCE_BACK"
    bc = {"boundary_conditions": [{"boundary_name": "WEST", "boundary_type": bc_name},
                                  {"boundary_name": "EAST", "boundary_type": bc_name},
                                  {"boundary_name": "SOUTH", "boundary_type": bc_name},
                                  {"boundary_name": "NORTH", "boundary_type": bc_name}]}
    mesh.setup_bc(bc)
    lattice.tau = np.inf  # no collision
    lattice.setup_streaming()
    f = np.copy(lattice.f0)
    # Streaming
    lattice.boundaries()
    lattice.streaming()
    # Wall cell
    cell = mesh.find_cell([2.5, 0.5])
    # Streaming
    for c in np.array([[0, 0], [1, 0], [-1, 0], [0, -1], [-1, -1], [1, -1]]):
        i = stencil.c_pop(c)
        assert f[i, cell.find_neighbour(-c).index] == approx(lattice.f[i, cell.index])
    # Bounce-back
    for c in np.array([[0, 1], [1, 1], [-1, 1]]):
        i = stencil.c_pop(c)
        assert f[stencil.i_opp[i], cell.index] == approx(lattice.f[i, cell.index])


def test_bounce_back_concave_corner_d2q9(mesh_box_2d):
    lattice = lattice_box(d=2, q=9, mesh=mesh_box_2d)
    stencil = lattice.stencil
    mesh = lattice.mesh
    bc_name = "BOUNCE_BACK"
    bc = {"boundary_conditions": [{"boundary_name": "WEST", "boundary_type": bc_name},
                                  {"boundary_name": "EAST", "boundary_type": bc_name},
                                  {"boundary_name": "SOUTH", "boundary_type": bc_name},
                                  {"boundary_name": "NORTH", "boundary_type": bc_name}]}
    mesh.setup_bc(bc)
    lattice.tau = np.inf  # no collision
    lattice.setup_streaming()
    f = np.copy(lattice.f0)
    # Streaming
    lattice.boundaries()
    lattice.streaming()
    # Concave corner cell
    cell = mesh.find_cell([0.5, 0.5])
    # Streaming
    for c in np.array([[0, 0], [-1, 0], [0, -1], [-1, -1]]):
        i = stencil.c_pop(c)
        assert f[i, cell.find_neighbour(-c).index] == approx(lattice.f[i, cell.index])
    # Bounce-back
    for c in np.array([[1, 0], [0, 1], [1, 1], [-1, 1], [1, -1]]):
        i = stencil.c_pop(c)
        assert f[stencil.i_opp[i], cell.index] == approx(lattice.f[i, cell.index])


def test_bounce_back_wall_data_wall_d2q9(mesh_box_2d):
    lattice = lattice_box(d=2, q=9, mesh=mesh_box_2d)
    stencil = lattice.stencil
    mesh = lattice.mesh
    bc_density = 1.0
    bc_velocity = [1.0, 2.0]
    bc_name = "BOUNCE_BACK"
    bc = {"boundary_conditions": [{"boundary_name": "WEST", "boundary_type": bc_name,
                                   "data": {"velocity": bc_velocity, "density": bc_density}},
                                  {"boundary_name": "EAST", "boundary_type": bc_name,
                                   "data": {"velocity": bc_velocity, "density": bc_density}},
                                  {"boundary_name": "SOUTH", "boundary_type": bc_name,
                                   "data": {"velocity": bc_velocity, "density": bc_density}},
                                  {"boundary_name": "NORTH", "boundary_type": bc_name,
                                   "data": {"velocity": bc_velocity, "density": bc_density}}]}
    mesh.setup_bc(bc)
    lattice.tau = np.inf  # no collision
    lattice.setup_streaming()
    f = np.copy(lattice.f0)
    # Streaming
    lattice.boundaries()
    lattice.streaming()
    # Fetch a boundary condition, they all have the same data
    bc = mesh.boundaries["WEST"].bc
    # Wall cell
    cell = mesh.find_cell([2.5, 0.5])
    # Streaming
    for c in np.array([[0, 0], [1, 0], [-1, 0], [0, -1], [-1, -1], [1, -1]]):
        i = stencil.c_pop(c)
        assert f[i, cell.find_neighbour(-c).index] == approx(lattice.f[i, cell.index])
    # Bounce-back
    for c in np.array([[0, 1], [1, 1], [-1, 1]]):
        i = stencil.c_pop(c)
        assert f[stencil.i_opp[i], cell.index] - bc.wall_data[i] == approx(lattice.f[i, cell.index])


def test_bounce_back_wall_data_concave_corner_d2q9(mesh_box_2d):
    lattice = lattice_box(d=2, q=9, mesh=mesh_box_2d)
    stencil = lattice.stencil
    mesh = lattice.mesh
    bc_density = 1.0
    bc_velocity = [1.0, 2.0]
    bc_name = "BOUNCE_BACK"
    bc = {"boundary_conditions": [{"boundary_name": "WEST", "boundary_type": bc_name,
                                   "data": {"velocity": bc_velocity, "density": bc_density}},
                                  {"boundary_name": "EAST", "boundary_type": bc_name,
                                   "data": {"velocity": bc_velocity, "density": bc_density}},
                                  {"boundary_name": "SOUTH", "boundary_type": bc_name,
                                   "data": {"velocity": bc_velocity, "density": bc_density}},
                                  {"boundary_name": "NORTH", "boundary_type": bc_name,
                                   "data": {"velocity": bc_velocity, "density": bc_density}}]}
    mesh.setup_bc(bc)
    lattice.tau = np.inf  # no collision
    lattice.setup_streaming()
    f = np.copy(lattice.f0)
    # Streaming
    lattice.boundaries()
    lattice.streaming()
    # Fetch a boundary condition, they all have the same data
    bc = mesh.boundaries["WEST"].bc
    # Concave corner cell
    cell = mesh.find_cell([0.5, 0.5])
    # Streaming
    for c in np.array([[0, 0], [-1, 0], [0, -1], [-1, -1]]):
        i = stencil.c_pop(c)
        assert f[i, cell.find_neighbour(-c).index] == approx(lattice.f[i, cell.index])
    # Bounce-back
    for c in np.array([[1, 0], [0, 1], [1, 1], [-1, 1], [1, -1]]):
        i = stencil.c_pop(c)
        assert f[stencil.i_opp[i], cell.index] - bc.wall_data[i] == approx(lattice.f[i, cell.index])


def test_anti_bounce_back_wall_d2q9(mesh_box_2d):
    lattice = lattice_box(d=2, q=9, mesh=mesh_box_2d)
    stencil = lattice.stencil
    mesh = lattice.mesh
    bc_density = 1.0
    bc_name = "ANTI_BOUNCE_BACK"
    bc = {"boundary_conditions": [{"boundary_name": "WEST", "boundary_type": bc_name,
                                   "data": {"density": bc_density}},
                                  {"boundary_name": "EAST", "boundary_type": bc_name,
                                   "data": {"density": bc_density}},
                                  {"boundary_name": "SOUTH", "boundary_type": bc_name,
                                   "data": {"density": bc_density}},
                                  {"boundary_name": "NORTH", "boundary_type": bc_name,
                                   "data": {"density": bc_density}}]}
    mesh.setup_bc(bc)
    lattice.tau = np.inf  # no collision
    lattice.setup_streaming()
    f = np.copy(lattice.f0)
    # Streaming
    lattice.boundaries()
    lattice.streaming()

    # Wall cell
    cell = mesh.find_cell([2.5, 0.5])
    cell_n = cell.find_neighbour([0, 1])
    wall_data = anti_bounce_back_wall_data(cell, cell_n, bc_density, lattice, stencil)
    # Streaming
    for c in np.array([[0, 0], [1, 0], [-1, 0], [0, -1], [-1, -1], [1, -1]]):
        i = stencil.c_pop(c)
        assert f[i, cell.find_neighbour(-c).index] == approx(lattice.f[i, cell.index])
    # Anti-Bounce-back
    for c in np.array([[0, 1], [1, 1], [-1, 1]]):
        i = stencil.c_pop(c)
        i_opp = stencil.i_opp[i]
        assert -f[i_opp, cell.index] + wall_data[i_opp] == approx(lattice.f[i, cell.index])


def test_anti_bounce_back_concave_corner_d2q9(mesh_box_2d):
    lattice = lattice_box(d=2, q=9, mesh=mesh_box_2d)
    stencil = lattice.stencil
    mesh = lattice.mesh
    bc_density = 1.0
    bc_name = "ANTI_BOUNCE_BACK"
    bc = {"boundary_conditions": [{"boundary_name": "WEST", "boundary_type": bc_name,
                                   "data": {"density": bc_density}},
                                  {"boundary_name": "EAST", "boundary_type": bc_name,
                                   "data": {"density": bc_density}},
                                  {"boundary_name": "SOUTH", "boundary_type": bc_name,
                                   "data": {"density": bc_density}},
                                  {"boundary_name": "NORTH", "boundary_type": bc_name,
                                   "data": {"density": bc_density}}]}
    mesh.setup_bc(bc)
    lattice.tau = np.inf  # no collision
    lattice.setup_streaming()
    f = np.copy(lattice.f0)
    # Streaming
    lattice.boundaries()
    lattice.streaming()
    # Concave corner cell
    cell = mesh.find_cell([0.5, 0.5])
    # Streaming
    for c in np.array([[0, 0], [-1, 0], [0, -1], [-1, -1]]):
        i = stencil.c_pop(c)
        assert f[i, cell.find_neighbour(-c).index] == approx(lattice.f[i, cell.index])
    # Symmetry - normal to X
    cell_n = cell.find_neighbour([1, 0])
    wall_data = anti_bounce_back_wall_data(cell, cell_n, bc_density, lattice, stencil)
    for c in np.array([[1, 0], [1, -1]]):
        i = stencil.c_pop(c)
        i_opp = stencil.i_opp[i]
        assert -f[i_opp, cell.index] + wall_data[i_opp] == approx(lattice.f[i, cell.index])
    # Symmetry - normal to Y
    cell_n = cell.find_neighbour([0, 1])
    wall_data = anti_bounce_back_wall_data(cell, cell_n, bc_density, lattice, stencil)
    for c in np.array([[0, 1], [-1, 1]]):
        i = stencil.c_pop(c)
        i_opp = stencil.i_opp[i]
        assert -f[i_opp, cell.index] + wall_data[i_opp] == approx(lattice.f[i, cell.index])
    # Symmetry - corner population
    cell_n = cell.find_neighbour([1, 1])
    wall_data = anti_bounce_back_wall_data(cell, cell_n, bc_density, lattice, stencil)
    for c in np.array([[1, 1]]):
        i = stencil.c_pop(c)
        i_opp = stencil.i_opp[i]
        assert -f[i_opp, cell.index] + wall_data[i_opp] == approx(lattice.f[i, cell.index])


def test_symmetry_wall_d2q9(mesh_box_2d):
    lattice = lattice_box(d=2, q=9, mesh=mesh_box_2d)
    stencil = lattice.stencil
    mesh = lattice.mesh
    bc_name = "SYMMETRY"
    bc = {"boundary_conditions": [{"boundary_name": "WEST", "boundary_type": bc_name},
                                  {"boundary_name": "EAST", "boundary_type": bc_name},
                                  {"boundary_name": "SOUTH", "boundary_type": bc_name},
                                  {"boundary_name": "NORTH", "boundary_type": bc_name}]}
    mesh.setup_bc(bc)
    lattice.tau = np.inf  # no collision
    lattice.setup_streaming()
    f = np.copy(lattice.f0)
    # Streaming
    lattice.boundaries()
    lattice.streaming()
    # Wall cell
    cell = mesh.find_cell([2.5, 0.5])
    normal = [0, -1]
    i_n = stencil.c_pop(normal)
    # Streaming
    for c in np.array([[0, 0], [1, 0], [-1, 0], [0, -1], [-1, -1], [1, -1]]):
        i = stencil.c_pop(c)
        assert f[i, cell.find_neighbour(-c).index] == approx(lattice.f[i, cell.index])
    # Symmetry
    for c in np.array([[0, 1], [1, 1], [-1, 1]]):
        i = stencil.c_pop(c)
        cell_s = cell.find_neighbour(-(c + normal))
        assert f[stencil.i_sym[i_n, i], cell_s.index] == approx(lattice.f[i, cell.index])


def test_symmetry_concave_corner_d2q9(mesh_box_2d):
    lattice = lattice_box(d=2, q=9, mesh=mesh_box_2d)
    stencil = lattice.stencil
    mesh = lattice.mesh
    bc_name = "SYMMETRY"
    bc = {"boundary_conditions": [{"boundary_name": "WEST", "boundary_type": bc_name},
                                  {"boundary_name": "EAST", "boundary_type": bc_name},
                                  {"boundary_name": "SOUTH", "boundary_type": bc_name},
                                  {"boundary_name": "NORTH", "boundary_type": bc_name}]}
    mesh.setup_bc(bc)
    lattice.tau = np.inf  # no collision
    lattice.setup_streaming()
    f = np.copy(lattice.f0)
    # Streaming
    lattice.boundaries()
    lattice.streaming()
    # Concave corner cell
    cell = mesh.find_cell([0.5, 0.5])
    # Streaming
    for c in np.array([[0, 0], [-1, 0], [0, -1], [-1, -1]]):
        i = stencil.c_pop(c)
        assert f[i, cell.find_neighbour(-c).index] == approx(lattice.f[i, cell.index])
    # Symmetry - normal to X
    normal = [-1, 0]
    i_n = stencil.c_pop(normal)
    for c in np.array([[1, 0], [1, -1]]):
        i = stencil.c_pop(c)
        cell_s = cell.find_neighbour(-(c + normal))
        assert f[stencil.i_sym[i_n, i], cell_s.index] == approx(lattice.f[i, cell.index])
    # Symmetry - normal to Y
    normal = [0, -1]
    i_n = stencil.c_pop(normal)
    for c in np.array([[0, 1], [-1, 1]]):
        i = stencil.c_pop(c)
        cell_s = cell.find_neighbour(-(c + normal))
        assert f[stencil.i_sym[i_n, i], cell_s.index] == approx(lattice.f[i, cell.index])
    # Symmetry - corner population
    normal = [-1, -1]
    for c in np.array([[1, 1]]):
        i_n = stencil.c_pop(normal)
        i = stencil.c_pop(c)
        assert f[stencil.i_sym[i_n, i], cell.index] == approx(lattice.f[i, cell.index])


def test_symmetry_meets_bounce_back_wall_d2q9(mesh_box_2d_corner_bc):
    lattice = lattice_box(d=2, q=9, mesh=mesh_box_2d_corner_bc)
    stencil = lattice.stencil
    mesh = lattice.mesh
    bc = {"boundary_conditions": [{"boundary_name": "SOUTHWEST", "boundary_type": "BOUNCE_BACK"},
                                  {"boundary_name": "NORTHWEST", "boundary_type": "BOUNCE_BACK"},
                                  {"boundary_name": "NORTHEAST", "boundary_type": "SYMMETRY"},
                                  {"boundary_name": "SOUTHEAST", "boundary_type": "SYMMETRY"}]}
    mesh.setup_bc(bc)
    lattice.tau = np.inf  # no collision
    lattice.setup_streaming()
    f = np.copy(lattice.f0)
    # Streaming
    lattice.boundaries()
    lattice.streaming()
    # Wall cell - above bounce-back
    cell = mesh.find_cell([2.5, 0.5])
    # Streaming
    for c in np.array([[0, 0], [1, 0], [-1, 0], [0, -1], [-1, -1], [1, -1]]):
        i = stencil.c_pop(c)
        assert f[i, cell.find_neighbour(-c).index] == approx(lattice.f[i, cell.index])
    # Bounce-back
    for c in np.array([[0, 1], [1, 1], [-1, 1]]):
        i = stencil.c_pop(c)
        assert f[stencil.i_opp[i], cell.index] == approx(lattice.f[i, cell.index])

    # Wall cell - above symmetry
    cell = mesh.find_cell([3.5, 0.5])
    normal = [0, -1]
    i_n = stencil.c_pop(normal)
    # Streaming
    for c in np.array([[0, 0], [1, 0], [-1, 0], [0, -1], [-1, -1], [1, -1]]):
        i = stencil.c_pop(c)
        assert f[i, cell.find_neighbour(-c).index] == approx(lattice.f[i, cell.index])
    # Symmetry
    for c in np.array([[0, 1], [-1, 1]]):
        i = stencil.c_pop(c)
        cell_s = cell.find_neighbour(-(c + normal))
        assert f[stencil.i_sym[i_n, i], cell_s.index] == approx(lattice.f[i, cell.index])
    # Delegated bounce-back
    for c in np.array([[1, 1]]):
        i = stencil.c_pop(c)
        assert f[stencil.i_opp[i], cell.index] == approx(lattice.f[i, cell.index])


def test_symmetry_meets_bounce_back_concave_corner_d2q9(mesh_box_2d):
    lattice = lattice_box(d=2, q=9, mesh=mesh_box_2d)
    stencil = lattice.stencil
    mesh = lattice.mesh
    bc_density = 1.0
    bc_velocity = [1.0, 2.0]
    bc = {"boundary_conditions": [{"boundary_name": "WEST", "boundary_type": "BOUNCE_BACK",
                                   "data": {"velocity": bc_velocity, "density": bc_density}},
                                  {"boundary_name": "EAST", "boundary_type": "BOUNCE_BACK"},
                                  {"boundary_name": "SOUTH", "boundary_type": "SYMMETRY"},
                                  {"boundary_name": "NORTH", "boundary_type": "SYMMETRY"}]}
    mesh.setup_bc(bc)
    lattice.tau = np.inf  # no collision
    lattice.setup_streaming()
    f = np.copy(lattice.f0)
    # Streaming
    lattice.boundaries()
    lattice.streaming()
    # Fetch a boundary condition, they all have the same data
    bc = mesh.boundaries["WEST"].bc
    # Concave corner cell - with wall data
    cell = mesh.find_cell([0.5, 0.5])
    # Streaming
    for c in np.array([[0, 0], [-1, 0], [0, -1], [-1, -1]]):
        i = stencil.c_pop(c)
        assert f[i, cell.find_neighbour(-c).index] == approx(lattice.f[i, cell.index])
    # Bounce-back
    for c in np.array([[1, 0], [1, 1], [1, -1]]):
        i = stencil.c_pop(c)
        assert f[stencil.i_opp[i], cell.index] - bc.wall_data[i] == approx(lattice.f[i, cell.index])
    # Symmetry - normal to Y
    normal = [0, -1]
    i_n = stencil.c_pop(normal)
    for c in np.array([[0, 1], [-1, 1]]):
        i = stencil.c_pop(c)
        cell_s = cell.find_neighbour(-(c + normal))
        assert f[stencil.i_sym[i_n, i], cell_s.index] == approx(lattice.f[i, cell.index])

    # Concave corner cell - no wall data
    cell = mesh.find_cell([5.5, 0.5])
    # Streaming
    for c in np.array([[0, 0], [1, 0], [0, -1], [1, -1]]):
        i = stencil.c_pop(c)
        assert f[i, cell.find_neighbour(-c).index] == approx(lattice.f[i, cell.index])
    # Bounce-back
    for c in np.array([[-1, 0], [-1, 1], [-1, -1]]):
        i = stencil.c_pop(c)
        assert f[stencil.i_opp[i], cell.index] == approx(lattice.f[i, cell.index])
    # Symmetry - normal to Y
    normal = [0, -1]
    i_n = stencil.c_pop(normal)
    for c in np.array([[0, 1], [1, 1]]):
        i = stencil.c_pop(c)
        cell_s = cell.find_neighbour(-(c + normal))
        assert f[stencil.i_sym[i_n, i], cell_s.index] == approx(lattice.f[i, cell.index])


def test_symmetry_meets_bounce_back_convex_corner_d2q9(mesh_box_2d_convex_corner):
    lattice = lattice_box(d=2, q=9, mesh=mesh_box_2d_convex_corner)
    stencil = lattice.stencil
    mesh = lattice.mesh
    bc = {"boundary_conditions": [{"boundary_name": "WEST", "boundary_type": "SYMMETRY"},
                                  {"boundary_name": "SOUTH", "boundary_type": "BOUNCE_BACK"},
                                  {"boundary_name": "WALL", "boundary_type": "ORPHAN"}]}
    mesh.setup_bc(bc)
    lattice.tau = np.inf  # no collision
    lattice.setup_streaming()
    f = np.copy(lattice.f0)
    # Streaming
    lattice.boundaries()
    lattice.streaming()
    # Concave corner cell
    cell = mesh.find_cell([0.5, 0.5])
    # Streaming
    for c in np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [-1, 1], [-1, -1], [1, -1]]):
        i = stencil.c_pop(c)
        assert f[i, cell.find_neighbour(-c).index] == approx(lattice.f[i, cell.index])
    # Bounce-back
    for c in np.array([[1, 1]]):
        i = stencil.c_pop(c)
        assert f[stencil.i_opp[i], cell.index] == approx(lattice.f[i, cell.index])

    # Cell normal to symmetry bc
    cell = mesh.find_cell([-0.5, 0.5])
    # Streaming
    for c in np.array([[0, 0], [1, 0], [-1, 0], [0, -1], [-1, 1], [-1, -1], [1, -1]]):
        i = stencil.c_pop(c)
        assert f[i, cell.find_neighbour(-c).index] == approx(lattice.f[i, cell.index])
    # Symmetry
    normal = [0, -1]
    i_n = stencil.c_pop(normal)
    for c in np.array([[0, 1], [1, 1]]):
        i = stencil.c_pop(c)
        cell_s = cell.find_neighbour(-(c + normal))
        assert f[stencil.i_sym[i_n, i], cell_s.index] == approx(lattice.f[i, cell.index])

    # Cell normal to bounce-back bc
    cell = mesh.find_cell([0.5, -0.5])
    # Streaming
    for c in np.array([[0, 0], [0, 1], [-1, 0], [0, -1], [-1, 1], [-1, -1], [1, -1]]):
        i = stencil.c_pop(c)
        assert f[i, cell.find_neighbour(-c).index] == approx(lattice.f[i, cell.index])
    # Bounce-back
    for c in np.array([[1, 0], [1, 1]]):
        i = stencil.c_pop(c)
        assert f[stencil.i_opp[i], cell.index] == approx(lattice.f[i, cell.index])


def test_bounce_back_wall_d3q19(mesh_box_3d):
    lattice = lattice_box(d=3, q=19, mesh=mesh_box_3d)
    stencil = lattice.stencil
    mesh = lattice.mesh
    bc_name = "BOUNCE_BACK"
    bc = {"boundary_conditions": [{"boundary_name": "WEST", "boundary_type": bc_name},
                                  {"boundary_name": "EAST", "boundary_type": bc_name},
                                  {"boundary_name": "SOUTH", "boundary_type": bc_name},
                                  {"boundary_name": "NORTH", "boundary_type": bc_name},
                                  {"boundary_name": "BACK", "boundary_type": bc_name},
                                  {"boundary_name": "FRONT", "boundary_type": bc_name}]}
    mesh.setup_bc(bc)
    lattice.tau = np.inf  # no collision
    lattice.setup_streaming()
    f = np.copy(lattice.f0)
    # Streaming
    lattice.boundaries()
    lattice.streaming()
    # Wall cell
    cell = mesh.find_cell([2.5, 0.5, 2.5])
    # Streaming
    for c in np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1],
                       [0, 0, -1], [-1, -1, 0], [1, 0, 1], [-1, 0, -1], [0, -1, -1],
                       [1, -1, 0], [1, 0, -1], [-1, 0, 1], [0, -1, 1]]):
        i = stencil.c_pop(c)
        assert f[i, cell.find_neighbour(-c).index] == approx(lattice.f[i, cell.index])
    # Bounce-back
    for c in np.array([[0, 1, 0], [1, 1, 0], [0, 1, 1], [-1, 1, 0], [0, 1, -1]]):
        i = stencil.c_pop(c)
        assert f[stencil.i_opp[i], cell.index] == approx(lattice.f[i, cell.index])


def test_bounce_back_wall_data_wall_d3q19(mesh_box_3d):
    lattice = lattice_box(d=3, q=19, mesh=mesh_box_3d)
    stencil = lattice.stencil
    mesh = lattice.mesh
    bc_velocity = [1.0, 2.0, 3.0]
    bc_density = 1.0
    bc_name = "BOUNCE_BACK"
    bc = {"boundary_conditions": [{"boundary_name": "WEST", "boundary_type": bc_name,
                                   "data": {"velocity": bc_velocity, "density": bc_density}},
                                  {"boundary_name": "EAST", "boundary_type": bc_name,
                                   "data": {"velocity": bc_velocity, "density": bc_density}},
                                  {"boundary_name": "SOUTH", "boundary_type": bc_name,
                                   "data": {"velocity": bc_velocity, "density": bc_density}},
                                  {"boundary_name": "NORTH", "boundary_type": bc_name,
                                   "data": {"velocity": bc_velocity, "density": bc_density}},
                                  {"boundary_name": "BACK", "boundary_type": bc_name,
                                   "data": {"velocity": bc_velocity, "density": bc_density}},
                                  {"boundary_name": "FRONT", "boundary_type": bc_name,
                                   "data": {"velocity": bc_velocity, "density": bc_density}}]}
    mesh.setup_bc(bc)
    lattice.tau = np.inf  # no collision
    lattice.setup_streaming()
    f = np.copy(lattice.f0)
    # Streaming
    lattice.boundaries()
    lattice.streaming()
    # Fetch a boundary condition, they all have the same data
    bc = mesh.boundaries["WEST"].bc
    # Wall cell
    cell = mesh.find_cell([2.5, 0.5, 2.5])
    # Streaming
    for c in np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1],
                       [0, 0, -1], [-1, -1, 0], [1, 0, 1], [-1, 0, -1], [0, -1, -1],
                       [1, -1, 0], [1, 0, -1], [-1, 0, 1], [0, -1, 1]]):
        i = stencil.c_pop(c)
        assert f[i, cell.find_neighbour(-c).index] == approx(lattice.f[i, cell.index])
    # Bounce-back
    for c in np.array([[0, 1, 0], [1, 1, 0], [0, 1, 1], [-1, 1, 0], [0, 1, -1]]):
        i = stencil.c_pop(c)
        assert f[stencil.i_opp[i], cell.index] - bc.wall_data[i] == approx(lattice.f[i, cell.index])


def test_anti_bounce_back_wall_d3q19(mesh_box_3d):
    lattice = lattice_box(d=3, q=19, mesh=mesh_box_3d)
    stencil = lattice.stencil
    mesh = lattice.mesh
    bc_density = 1.0
    bc_name = "ANTI_BOUNCE_BACK"
    bc = {"boundary_conditions": [{"boundary_name": "WEST", "boundary_type": bc_name,
                                   "data": {"density": bc_density}},
                                  {"boundary_name": "EAST", "boundary_type": bc_name,
                                   "data": {"density": bc_density}},
                                  {"boundary_name": "SOUTH", "boundary_type": bc_name,
                                   "data": {"density": bc_density}},
                                  {"boundary_name": "NORTH", "boundary_type": bc_name,
                                   "data": {"density": bc_density}},
                                  {"boundary_name": "FRONT", "boundary_type": bc_name,
                                   "data": {"density": bc_density}},
                                  {"boundary_name": "BACK", "boundary_type": bc_name,
                                   "data": {"density": bc_density}}]}
    mesh.setup_bc(bc)
    lattice.tau = np.inf  # no collision
    lattice.setup_streaming()
    f = np.copy(lattice.f0)
    # Streaming
    lattice.boundaries()
    lattice.streaming()

    # Wall cell
    cell = mesh.find_cell([2.5, 0.5, 2.5])
    cell_n = cell.find_neighbour([0, 1, 0])
    wall_data = anti_bounce_back_wall_data(cell, cell_n, bc_density, lattice, stencil)
    # Streaming
    for c in np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1],
                       [0, 0, -1], [-1, -1, 0], [1, 0, 1], [-1, 0, -1], [0, -1, -1],
                       [1, -1, 0], [1, 0, -1], [-1, 0, 1], [0, -1, 1]]):
        i = stencil.c_pop(c)
        assert f[i, cell.find_neighbour(-c).index] == approx(lattice.f[i, cell.index])
    # Bounce-back
    for c in np.array([[0, 1, 0], [1, 1, 0], [0, 1, 1], [-1, 1, 0], [0, 1, -1]]):
        i = stencil.c_pop(c)
        i_opp = stencil.i_opp[i]
        assert -f[i_opp, cell.index] + wall_data[i_opp] == approx(lattice.f[i, cell.index])


def test_symmetry_wall_d3q19(mesh_box_3d):
    lattice = lattice_box(d=3, q=19, mesh=mesh_box_3d)
    stencil = lattice.stencil
    mesh = lattice.mesh
    bc_name = "SYMMETRY"
    bc = {"boundary_conditions": [{"boundary_name": "WEST", "boundary_type": bc_name},
                                  {"boundary_name": "EAST", "boundary_type": bc_name},
                                  {"boundary_name": "SOUTH", "boundary_type": bc_name},
                                  {"boundary_name": "NORTH", "boundary_type": bc_name},
                                  {"boundary_name": "BACK", "boundary_type": bc_name},
                                  {"boundary_name": "FRONT", "boundary_type": bc_name}]}
    mesh.setup_bc(bc)
    lattice.tau = np.inf  # no collision
    lattice.setup_streaming()
    f = np.copy(lattice.f0)
    # Streaming
    lattice.boundaries()
    lattice.streaming()
    # Wall cell
    cell = mesh.find_cell([2.5, 0.5, 2.5])
    normal = [0, -1, 0]
    i_n = stencil.c_pop(normal)
    # Streaming
    for c in np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1],
                       [0, 0, -1], [-1, -1, 0], [1, 0, 1], [-1, 0, -1], [0, -1, -1],
                       [1, -1, 0], [1, 0, -1], [-1, 0, 1], [0, -1, 1]]):
        i = stencil.c_pop(c)
        assert f[i, cell.find_neighbour(-c).index] == approx(lattice.f[i, cell.index])
    # Symmetry
    for c in np.array([[0, 1, 0], [1, 1, 0], [0, 1, 1], [-1, 1, 0], [0, 1, -1]]):
        i = stencil.c_pop(c)
        cell_s = cell.find_neighbour(-(c + normal))
        assert f[stencil.i_sym[i_n, i], cell_s.index] == approx(lattice.f[i, cell.index])
