import pytest
from pytest import approx
from lbm.mesh.mesh_generator import MeshGenerator


@pytest.fixture
def mesh_cross_2d():
    n = 3
    mesh_gen = MeshGenerator()
    # Center block
    boundary_names = {"west": None, "east": None, "south": None, "north": None}
    mesh_gen.create_block(nx=n, ny=n, x_offset=0, y_offset=0, boundary_names=boundary_names)
    # East block
    boundary_names = {"west": None, "east": "WALL", "south": "WALL", "north": "WALL"}
    mesh_gen.create_block(nx=n, ny=n, x_offset=n, y_offset=0, boundary_names=boundary_names)
    # West block
    boundary_names = {"west": "WALL", "east": None, "south": "WALL", "north": "WALL"}
    mesh_gen.create_block(nx=n, ny=n, x_offset=-n, y_offset=0, boundary_names=boundary_names)
    # North block
    boundary_names = {"west": "WALL", "east": "WALL", "south": None, "north": "WALL"}
    mesh_gen.create_block(nx=n, ny=n, x_offset=0, y_offset=n, boundary_names=boundary_names)
    # South block
    boundary_names = {"west": "WALL", "east": "WALL", "south": "WALL", "north": None}
    mesh_gen.create_block(nx=n, ny=n, x_offset=0, y_offset=-n, boundary_names=boundary_names)

    mesh = mesh_gen.mesh
    mesh.setup()
    return mesh


@pytest.fixture
def mesh_cross_3d():
    n = 3
    mesh_gen = MeshGenerator()
    # Center block
    boundary_names = {"west": None, "east": None, "south": None, "north": None, "front": None, "back": None}
    mesh_gen.create_block(nx=n, ny=n, nz=n, x_offset=0, y_offset=0, z_offset=0, boundary_names=boundary_names)
    # East block
    boundary_names = {"west": None, "east": "WALL", "south": "WALL", "north": "WALL", "back": "WALL", "front": "WALL"}
    mesh_gen.create_block(nx=n, ny=n, nz=n, x_offset=n, y_offset=0, z_offset=0, boundary_names=boundary_names)
    # West block
    boundary_names = {"west": "WALL", "east": None, "south": "WALL", "north": "WALL", "back": "WALL", "front": "WALL"}
    mesh_gen.create_block(nx=n, ny=n, nz=n, x_offset=-n, y_offset=0, z_offset=0, boundary_names=boundary_names)
    # North block
    boundary_names = {"west": "WALL", "east": "WALL", "south": None, "north": "WALL", "back": "WALL", "front": "WALL"}
    mesh_gen.create_block(nx=n, ny=n, nz=n, x_offset=0, y_offset=n, z_offset=0, boundary_names=boundary_names)
    # South block
    boundary_names = {"west": "WALL", "east": "WALL", "south": "WALL", "north": None, "back": "WALL", "front": "WALL"}
    mesh_gen.create_block(nx=n, ny=n, nz=n, x_offset=0, y_offset=-n, z_offset=0, boundary_names=boundary_names)
    # Front block
    boundary_names = {"west": "WALL", "east": "WALL", "south": "WALL", "north": "WALL", "back": None, "front": "WALL"}
    mesh_gen.create_block(nx=n, ny=n, nz=n, x_offset=0, y_offset=0, z_offset=n, boundary_names=boundary_names)
    # Back block
    boundary_names = {"west": "WALL", "east": "WALL", "south": "WALL", "north": "WALL", "back": "WALL", "front": None}
    mesh_gen.create_block(nx=n, ny=n, nz=n, x_offset=0, y_offset=0, z_offset=-n, boundary_names=boundary_names)

    mesh = mesh_gen.mesh
    mesh.setup()
    return mesh


def test_find_node_2d(mesh_cross_2d):
    coordinate = [-100, -100]
    node = mesh_cross_2d.find_node(coordinate)
    assert node is None

    coordinate = [0.0, 0.0]
    node = mesh_cross_2d.find_node(coordinate)
    assert node.r[0] == approx(coordinate[0])
    assert node.r[1] == approx(coordinate[1])

    coordinate = [3.0, 3.0]
    node = mesh_cross_2d.find_node(coordinate)
    assert node.r[0] == approx(coordinate[0])
    assert node.r[1] == approx(coordinate[1])


def test_find_face_2d(mesh_cross_2d):
    coordinate = [-100, -100]
    face = mesh_cross_2d.find_face(coordinate)
    assert face is None

    coordinate = [-3.0, 0.5]
    face = mesh_cross_2d.find_face(coordinate)
    assert face.coordinate[0] == approx(coordinate[0])
    assert face.coordinate[1] == approx(coordinate[1])

    coordinate = [0.5, -3.0]
    face = mesh_cross_2d.find_face(coordinate)
    assert face.coordinate[0] == approx(coordinate[0])
    assert face.coordinate[1] == approx(coordinate[1])


def test_find_cell_2d(mesh_cross_2d):
    coordinate = [-100, -100]
    cell = mesh_cross_2d.find_cell(coordinate)
    assert cell is None

    coordinate = [0.5, 0.5]
    cell = mesh_cross_2d.find_cell(coordinate)
    assert cell.coordinate[0] == approx(coordinate[0])
    assert cell.coordinate[1] == approx(coordinate[1])

    coordinate = [2.5, 2.5]
    cell = mesh_cross_2d.find_cell(coordinate)
    assert cell.coordinate[0] == approx(coordinate[0])
    assert cell.coordinate[1] == approx(coordinate[1])


def test_face_neighbours_2d(mesh_cross_2d):
    coordinate = [-3.0, 0.5]
    face1 = mesh_cross_2d.find_face(coordinate)

    coordinate = [2.5, 6.0]
    face2 = mesh_cross_2d.find_face(coordinate)

    coordinate = [-2.5, 0.5]
    cell1 = mesh_cross_2d.find_cell(coordinate)

    coordinate = [2.5, 5.5]
    cell2 = mesh_cross_2d.find_cell(coordinate)

    assert face1.neighbours[0] == cell1
    assert cell1.neighbours[0] == face1
    assert face2.neighbours[0] == cell2
    assert cell2.neighbours[3] == face2


def test_cell_neighbours_2d(mesh_cross_2d):
    coordinate = [4.5, 1.5]
    cell = mesh_cross_2d.find_cell(coordinate)
    coordinate = cell.coordinate

    cell_n = cell.neighbours[0]
    assert cell_n.coordinate[0] == approx(coordinate[0] - 1.0)
    assert cell_n.coordinate[1] == approx(coordinate[1] - 0.0)

    cell_n = cell.neighbours[1]
    assert cell_n.coordinate[0] == approx(coordinate[0] + 1.0)
    assert cell_n.coordinate[1] == approx(coordinate[1] - 0.0)

    cell_n = cell.neighbours[2]
    assert cell_n.coordinate[0] == approx(coordinate[0] - 0.0)
    assert cell_n.coordinate[1] == approx(coordinate[1] - 1.0)

    cell_n = cell.neighbours[3]
    assert cell_n.coordinate[0] == approx(coordinate[0] - 0.0)
    assert cell_n.coordinate[1] == approx(coordinate[1] + 1.0)


def test_find_cell_neighbours_zero_2d(mesh_cross_2d):
    coordinate = [4.5, 1.5]
    cell = mesh_cross_2d.find_cell(coordinate)
    cell_n = cell.find_neighbour([0, 0])
    assert cell_n == cell


def test_find_cell_neighbours_cardinal_2d(mesh_cross_2d):
    coordinate = [4.5, 1.5]
    cell = mesh_cross_2d.find_cell(coordinate)

    cell_n = cell.find_neighbour([-1, 0])
    assert cell_n == cell.neighbours[0]

    cell_n = cell.find_neighbour([1, 0])
    assert cell_n == cell.neighbours[1]

    cell_n = cell.find_neighbour([0, -1])
    assert cell_n == cell.neighbours[2]

    cell_n = cell.find_neighbour([0, 1])
    assert cell_n == cell.neighbours[3]


def test_find_cell_neighbours_none_existing_2d(mesh_cross_2d):
    coordinate = [-2.5, 0.5]
    cell = mesh_cross_2d.find_cell(coordinate)
    cell_n = cell.find_neighbour([-1, -1])
    assert cell_n == []


def test_find_cell_neighbours_distanced_2d(mesh_cross_2d):
    coordinate = [-1.5, 1.5]
    cell = mesh_cross_2d.find_cell(coordinate)

    cell_n = cell.find_neighbour([1, 1])
    assert cell_n == mesh_cross_2d.find_cell([-0.5, 2.5])

    cell_n = cell.find_neighbour([3, -3])
    assert cell_n == mesh_cross_2d.find_cell([1.5, -1.5])

    cell_n = cell.find_neighbour([3, 4])
    assert cell_n == mesh_cross_2d.find_cell([1.5, 5.5])


def test_find_cell_neighbours_face_2d(mesh_cross_2d):
    coordinate = [-2.5, 0.5]
    cell = mesh_cross_2d.find_cell(coordinate)

    element_list = cell.find_neighbour([-1, 0])
    face = mesh_cross_2d.find_face([-3.0, 0.5])
    assert len(element_list) == 1
    assert face in element_list


def test_find_cell_neighbours_face_list_2d(mesh_cross_2d):
    coordinate = [0.5, 0.5]
    cell = mesh_cross_2d.find_cell(coordinate)

    element_list = cell.find_neighbour([-1, -1])
    face1 = mesh_cross_2d.find_face([-0.5, 0.0])
    face2 = mesh_cross_2d.find_face([0.0, -0.5])
    assert len(element_list) == 2
    assert face1 in element_list
    assert face2 in element_list


def test_find_node_3d(mesh_cross_3d):
    coordinate = [-100, -100, -100]
    node = mesh_cross_3d.find_node(coordinate)
    assert node is None

    coordinate = [0.0, 0.0, 0.0]
    node = mesh_cross_3d.find_node(coordinate)
    assert node.r[0] == approx(coordinate[0])
    assert node.r[1] == approx(coordinate[1])
    assert node.r[2] == approx(coordinate[2])

    coordinate = [3.0, 3.0, 3.0]
    node = mesh_cross_3d.find_node(coordinate)
    assert node.r[0] == approx(coordinate[0])
    assert node.r[1] == approx(coordinate[1])
    assert node.r[2] == approx(coordinate[2])


def test_find_face_3d(mesh_cross_3d):
    coordinate = [-100, -100, -100]
    face = mesh_cross_3d.find_face(coordinate)
    assert face is None

    coordinate = [-3.0, 0.5, 0.5]
    face = mesh_cross_3d.find_face(coordinate)
    assert face.coordinate[0] == approx(coordinate[0])
    assert face.coordinate[1] == approx(coordinate[1])
    assert face.coordinate[2] == approx(coordinate[2])

    coordinate = [0.5, -3.0, 2.5]
    face = mesh_cross_3d.find_face(coordinate)
    assert face.coordinate[0] == approx(coordinate[0])
    assert face.coordinate[1] == approx(coordinate[1])
    assert face.coordinate[2] == approx(coordinate[2])


def test_find_cell_3d(mesh_cross_3d):
    coordinate = [-100, -100, -100]
    cell = mesh_cross_3d.find_cell(coordinate)
    assert cell is None

    coordinate = [0.5, 0.5, 0.5]
    cell = mesh_cross_3d.find_cell(coordinate)
    assert cell.coordinate[0] == approx(coordinate[0])
    assert cell.coordinate[1] == approx(coordinate[1])
    assert cell.coordinate[2] == approx(coordinate[2])

    coordinate = [2.5, 2.5, 2.5]
    cell = mesh_cross_3d.find_cell(coordinate)
    assert cell.coordinate[0] == approx(coordinate[0])
    assert cell.coordinate[1] == approx(coordinate[1])
    assert cell.coordinate[2] == approx(coordinate[2])


def test_face_neighbours_3d(mesh_cross_3d):
    coordinate = [-3.0, 0.5, 0.5]
    face1 = mesh_cross_3d.find_face(coordinate)

    coordinate = [0.5, -3.0, 2.5]
    face2 = mesh_cross_3d.find_face(coordinate)

    coordinate = [-2.5, 0.5, 0.5]
    cell1 = mesh_cross_3d.find_cell(coordinate)

    coordinate = [0.5, -2.5, 2.5]
    cell2 = mesh_cross_3d.find_cell(coordinate)

    assert face1.neighbours[0] == cell1
    assert cell1.neighbours[0] == face1
    assert face2.neighbours[0] == cell2
    assert cell2.neighbours[2] == face2


def test_cell_neighbours_3d(mesh_cross_3d):
    coordinate = [2.5, 2.5, 2.5]
    cell = mesh_cross_3d.find_cell(coordinate)
    coordinate = cell.coordinate

    cell_n = cell.neighbours[0]
    assert cell_n.coordinate[0] == approx(coordinate[0] - 1.0)
    assert cell_n.coordinate[1] == approx(coordinate[1] - 0.0)
    assert cell_n.coordinate[2] == approx(coordinate[2] - 0.0)

    cell_n = cell.neighbours[1]
    assert cell_n.coordinate[0] == approx(coordinate[0] + 1.0)
    assert cell_n.coordinate[1] == approx(coordinate[1] - 0.0)
    assert cell_n.coordinate[2] == approx(coordinate[2] - 0.0)

    cell_n = cell.neighbours[2]
    assert cell_n.coordinate[0] == approx(coordinate[0] - 0.0)
    assert cell_n.coordinate[1] == approx(coordinate[1] - 1.0)
    assert cell_n.coordinate[2] == approx(coordinate[2] - 0.0)

    cell_n = cell.neighbours[3]
    assert cell_n.coordinate[0] == approx(coordinate[0] - 0.0)
    assert cell_n.coordinate[1] == approx(coordinate[1] + 1.0)
    assert cell_n.coordinate[2] == approx(coordinate[2] - 0.0)

    cell_n = cell.neighbours[4]
    assert cell_n.coordinate[0] == approx(coordinate[0] - 0.0)
    assert cell_n.coordinate[1] == approx(coordinate[1] - 0.0)
    assert cell_n.coordinate[2] == approx(coordinate[2] - 1.0)

    cell_n = cell.neighbours[5]
    assert cell_n.coordinate[0] == approx(coordinate[0] - 0.0)
    assert cell_n.coordinate[1] == approx(coordinate[1] + 0.0)
    assert cell_n.coordinate[2] == approx(coordinate[2] + 1.0)


def test_find_cell_neighbours_zero_3d(mesh_cross_3d):
    coordinate = [2.5, 2.5, 2.5]
    cell = mesh_cross_3d.find_cell(coordinate)
    cell_n = cell.find_neighbour([0, 0, 0])
    assert cell_n == cell


def test_find_cell_neighbours_cardinal_3d(mesh_cross_3d):
    coordinate = [2.5, 2.5, 2.5]
    cell = mesh_cross_3d.find_cell(coordinate)

    cell_n = cell.find_neighbour([-1, 0, 0])
    assert cell_n == cell.neighbours[0]

    cell_n = cell.find_neighbour([1, 0, 0])
    assert cell_n == cell.neighbours[1]

    cell_n = cell.find_neighbour([0, -1, 0])
    assert cell_n == cell.neighbours[2]

    cell_n = cell.find_neighbour([0, 1, 0])
    assert cell_n == cell.neighbours[3]

    cell_n = cell.find_neighbour([0, 0, -1])
    assert cell_n == cell.neighbours[4]

    cell_n = cell.find_neighbour([0, 0, 1])
    assert cell_n == cell.neighbours[5]


def test_find_cell_neighbours_none_existing_3d(mesh_cross_3d):
    coordinate = [-2.5, 0.5, 0.5]
    cell = mesh_cross_3d.find_cell(coordinate)
    cell_n = cell.find_neighbour([-1, -1, -1])
    assert cell_n == []


def test_find_cell_neighbours_distanced_3d(mesh_cross_3d):
    coordinate = [-2.5, 1.5, 1.5]
    cell = mesh_cross_3d.find_cell(coordinate)

    cell_n = cell.find_neighbour([1, 1, 1])
    assert cell_n == mesh_cross_3d.find_cell([-1.5, 2.5, 2.5])

    cell_n = cell.find_neighbour([4, -3, 0])
    assert cell_n == mesh_cross_3d.find_cell([1.5, -1.5, 1.5])

    cell_n = cell.find_neighbour([4, 0, 3])
    assert cell_n == mesh_cross_3d.find_cell([1.5, 1.5, 4.5])


def test_find_cell_neighbours_face_3d(mesh_cross_3d):
    coordinate = [-2.5, 1.5, 1.5]
    cell = mesh_cross_3d.find_cell(coordinate)

    element_list = cell.find_neighbour([-1, 0, 0])
    face = mesh_cross_3d.find_face([-3.0, 1.5, 1.5])
    assert len(element_list) == 1
    assert face in element_list


def test_find_cell_neighbours_face_list_3d(mesh_cross_3d):
    coordinate = [0.5, 0.5, 0.5]
    cell = mesh_cross_3d.find_cell(coordinate)
    print(cell, cell.coordinate)

    element_list = cell.find_neighbour([-1, -1, 0])
    face1 = mesh_cross_3d.find_face([-0.5, 0.0, 0.5])
    face2 = mesh_cross_3d.find_face([0.0, -0.5, 0.5])
    assert len(element_list) == 2
    assert face1 in element_list
    assert face2 in element_list

    element_list = cell.find_neighbour([-1, 0, -1])
    face1 = mesh_cross_3d.find_face([-0.5, 0.5, 0.0])
    face2 = mesh_cross_3d.find_face([0.0, 0.5, -0.5])
    assert len(element_list) == 2
    assert face1 in element_list
    assert face2 in element_list

    element_list = cell.find_neighbour([0, -1, -1])
    face1 = mesh_cross_3d.find_face([0.5, -0.5, 0.0])
    face2 = mesh_cross_3d.find_face([0.5, 0.0, -0.5])
    assert len(element_list) == 2
    assert face1 in element_list
    assert face2 in element_list
