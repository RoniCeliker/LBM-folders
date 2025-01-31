import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lbm.simulation import Simulation
from lbm.core.stencil import Stencil
from lbm.core.lattice import Lattice
from lbm.mesh.mesh import read_ffa_mesh
from lbm.mesh.mesh_generator import MeshGenerator


def main():
    # Parameters
    if len(sys.argv) < 2:
        output_dir = "output/tmp"
    else:
        output_dir = sys.argv[1]
    app_name = "shear_layer_2d"
    stencil = Stencil(d=2, q=9)
    Re = 2000    # Reynolds number
    Ma = 0.1    # Mach number
    height = 200
    plate_length = 150

    velocity1 = [Ma * stencil.cs, 0.0]
    velocity2 = [0.5 * Ma * stencil.cs, 0.0]
    velocity = [0.75 * Ma * stencil.cs, 0.0]
    density = 1.0

    bc = create_bc(velocity1=velocity1, velocity2=velocity2, density=density)
    nu = height * velocity1[0] / Re
    tau = nu / stencil.cs ** 2 + 0.5
    print(f"tau = {tau}")

    print("Creating mesh")
    mesh = create_mesh(height=height, plate_length=plate_length)
    # # Write mesh to file
    # mesh.write_ffa(f"{app_name}.bmsh")
    mesh.setup()
    # # Write mesh cell-cell connectivity
    # mesh.write_ffa_connectivity(f"{app_name}.bcon")

    # # Read mesh from file
    # mesh = read_ffa_mesh(f"{app_name}.bmsh")
    # # Read mesh cell-cell connectivity
    # mesh.read_ffa_connectivity(f"{app_name}.bcon")
    
    mesh.setup_bc(bc)

    # Create lattice
    lattice = Lattice(stencil=stencil, mesh=mesh)
    lattice.tau = tau
    lattice.u[:] = velocity
    # Setup lattice streaming maps
    lattice.setup_streaming()
    # # Setup lattice streaming maps in parallel
    # lattice.setup_streaming(mesh=f"{app_name}.bmsh", connectivity=f"{app_name}.bcon", bc=bc, processes=4)
    # # Write lattice streaming maps to file
    # lattice.write_ffa_streaming_maps(f"{app_name}.bmap")
    # # Read lattice streaming maps from file
    # lattice.read_ffa_streaming_maps(f"{app_name}.bmap")

    sim = Simulation(lattice=lattice,
                     output_directory=output_dir,
                     application_name=app_name)
    sim.run(max_it=40000, mod_it=100)


def create_bc(velocity1: list[float], velocity2: list[float], density: float):
    bc = {"boundary_conditions": [{"boundary_name": "INLET_TOP", "boundary_type": "BOUNCE_BACK",
                                   "data": {"velocity": velocity1, "density": density}},
                                  {"boundary_name": "INLET_BOT", "boundary_type": "BOUNCE_BACK",
                                   "data": {"velocity": velocity2, "density": density}},
                                  {"boundary_name": "OUTLET", "boundary_type": "ANTI_BOUNCE_BACK",
                                   "data": {"density": density}},
                                  {"boundary_name": "TOP", "boundary_type": "SYMMETRY"},
                                  {"boundary_name": "BOT", "boundary_type": "SYMMETRY"},
                                  {"boundary_name": "PLATE_TOP", "boundary_type": "SYMMETRY"},
                                  {"boundary_name": "PLATE_BOT", "boundary_type": "SYMMETRY"}]}
    return bc


def create_mesh(height: int, plate_length: int):
    mesh_gen = MeshGenerator()

    # East
    boundary_names = {"west": None, "east": "OUTLET", "south": "BOT", "north": "TOP"}
    mesh_gen.create_block(nx=3*plate_length, ny=2*height, x_offset=plate_length, y_offset=-height,
                          boundary_names=boundary_names, renumber=False)
    # North west
    boundary_names = {"west": "INLET_TOP", "east": None, "south": "PLATE_TOP", "north": "TOP"}
    mesh_gen.create_block(nx=plate_length, ny=height, x_offset=0, y_offset=0,
                          boundary_names=boundary_names, renumber=False)
    # South west
    boundary_names = {"west": "INLET_BOT", "east": None, "south": "BOT", "north": "PLATE_BOT"}
    mesh_gen.create_block(nx=plate_length, ny=height, x_offset=0, y_offset=-height,
                          boundary_names=boundary_names, renumber=False)

    mesh_gen.renumber_all()
    return mesh_gen.mesh


if __name__ == "__main__":
    main()
