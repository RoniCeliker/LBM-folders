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
    app_name = "lid_driven_cavity_2d"
    stencil = Stencil(d=2, q=9)
    Re = 100   # Reynolds number
    Ma = 0.1    # Mach number
    L = 100     # Characteristic length

    velocity = [Ma * stencil.cs, 0.0]
    density = 1.0

    nu = L * velocity[0] / Re
    tau = nu / stencil.cs ** 2 + 0.5
    print(f"tau = {tau}")

# Create mesh
    mesh = create_mesh(L=L)
    # # Write mesh to file
    # mesh.write_ffa(f"{app_name}.bmsh")
    mesh.setup()
    # # Write mesh cell-cell connectivity
    # mesh.write_ffa_connectivity(f"{app_name}.bcon")

    # # Read mesh from file
    # mesh = read_ffa_mesh(f"{app_name}.bmsh")
    # # Read mesh cell-cell connectivity
    # mesh.read_ffa_connectivity(f"{app_name}.bcon")
    
    bc = create_bc(velocity=velocity, density=density)
    mesh.setup_bc(bc)

    # Create lattice
    lattice = Lattice(stencil=stencil, mesh=mesh)
    lattice.tau = tau
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
    sim.run(max_it=5000, mod_it=100)


def create_bc(velocity: list[float], density: float):
    bc = {"boundary_conditions": [{"boundary_name": "MOVING_WALL", "boundary_type": "BOUNCE_BACK",
                                   "data": {"velocity": velocity, "density": density}},
                                  {"boundary_name": "RESTING_WALL", "boundary_type": "BOUNCE_BACK"}]}
    return bc


def create_mesh(L: int):
    mesh_gen = MeshGenerator()

    boundary_names = {"west": "RESTING_WALL", "east": "RESTING_WALL", "south": "RESTING_WALL", "north": "MOVING_WALL"}
    mesh_gen.create_block(nx=L, ny=L,
                          boundary_names=boundary_names, renumber=False)
    mesh_gen.renumber_all()
    return mesh_gen.mesh


if __name__ == "__main__":
    main()
