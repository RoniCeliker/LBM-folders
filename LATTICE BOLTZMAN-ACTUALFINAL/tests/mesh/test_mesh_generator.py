import pytest
from pytest import approx
from lbm.mesh.mesh_generator import MeshGenerator


def test_normals_2d():
    n = 3
    mesh_gen = MeshGenerator()
    boundary_names = {"west": "WEST", "east": "EAST",
                      "south": "SOUTH", "north": "NORTH"}
    mesh_gen.create_block(nx=n, ny=n, boundary_names=boundary_names)
    mesh = mesh_gen.mesh
    for face in mesh.boundaries["WEST"].faces:
        normal = face.normal
        assert normal[0] == approx(-1.0)
        assert normal[1] == approx(0.0)
    for face in mesh.boundaries["EAST"].faces:
        normal = face.normal
        assert normal[0] == approx(1.0)
        assert normal[1] == approx(0.0)
    for face in mesh.boundaries["SOUTH"].faces:
        normal = face.normal
        assert normal[0] == approx(0.0)
        assert normal[1] == approx(-1.0)
    for face in mesh.boundaries["NORTH"].faces:
        normal = face.normal
        assert normal[0] == approx(0.0)
        assert normal[1] == approx(1.0)


def test_normals_3d():
    n = 3
    mesh_gen = MeshGenerator()
    boundary_names = {"west": "WEST", "east": "EAST",
                      "south": "SOUTH", "north": "NORTH",
                      "back": "BACK", "front": "FRONT"}
    mesh_gen.create_block(nx=n, ny=n, nz=n, boundary_names=boundary_names)
    mesh = mesh_gen.mesh
    for face in mesh.boundaries["WEST"].faces:
        normal = face.normal
        assert normal[0] == approx(-1.0)
        assert normal[1] == approx(0.0)
        assert normal[2] == approx(0.0)
    for face in mesh.boundaries["EAST"].faces:
        normal = face.normal
        assert normal[0] == approx(1.0)
        assert normal[1] == approx(0.0)
        assert normal[2] == approx(0.0)
    for face in mesh.boundaries["SOUTH"].faces:
        normal = face.normal
        assert normal[0] == approx(0.0)
        assert normal[1] == approx(-1.0)
        assert normal[2] == approx(0.0)
    for face in mesh.boundaries["NORTH"].faces:
        normal = face.normal
        assert normal[0] == approx(0.0)
        assert normal[1] == approx(1.0)
        assert normal[2] == approx(0.0)
    for face in mesh.boundaries["BACK"].faces:
        normal = face.normal
        assert normal[0] == approx(0.0)
        assert normal[1] == approx(0.0)
        assert normal[2] == approx(-1.0)
    for face in mesh.boundaries["FRONT"].faces:
        normal = face.normal
        assert normal[0] == approx(0.0)
        assert normal[1] == approx(0.0)
        assert normal[2] == approx(1.0)
