import sys
import os
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
    os.makedirs(output_dir, exist_ok=True)
    app_name = "cavity_3d"
    stencil = Stencil(d=3, q=19)
    Re = 250    # Reynolds number
    Ma = 0.1    # Mach number
    L = 50
    Lx = 1.0
    Ly = 1.0
    Lz = 1.5
    Lbx = 1.0
    Lby = 0.2
    Lbz = 0.3
    velocity = [Ma * stencil.cs, 0.0, 0.0]
    density = 1.0

    bc = create_bc(velocity=velocity, density=density)
    nu = L*Lbx * velocity[0] / Re
    tau = nu / stencil.cs ** 2 + 0.5
    print(f"tau = {tau}")

    # Create mesh
    mesh = create_mesh(L, Lx, Ly, Lz, Lbx, Lby, Lbz)
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
    sim.run(max_it=50000, mod_it=100)


def create_bc(velocity: list[float], density: float):
    bc = {"boundary_conditions": [{"boundary_name": "INLET", "boundary_type": "BOUNCE_BACK",
                                   "data": {"velocity": velocity, "density": density}},
                                  {"boundary_name": "OUTLET", "boundary_type": "ANTI_BOUNCE_BACK",
                                   "data": {"density": density}},
                                  {"boundary_name": "CAVITY", "boundary_type": "BOUNCE_BACK"},
                                  {"boundary_name": "TOP", "boundary_type": "SYMMETRY"},
                                  {"boundary_name": "BOT", "boundary_type": "SYMMETRY"},
                                  {"boundary_name": "WALL", "boundary_type": "SYMMETRY"}]}
    return bc


def create_mesh(L, Lx, Ly, Lz, Lbx, Lby, Lbz):
    mesh_gen = MeshGenerator()
    # West
    boundary_names = {"west": "INLET", "east": None, "south": "BOT", "north": "TOP", "back": "WALL", "front": "WALL"}
    mesh_gen.create_block(nx=int(L*Lx), ny=int(L*Ly), nz=int(L*Lz),
                          x_offset=0, y_offset=0, z_offset=0,
                          boundary_names=boundary_names, renumber=False, replace_nodes=False)
    # East
    boundary_names = {"west": None, "east": "OUTLET", "south": "BOT", "north": "TOP", "back": "WALL", "front": "WALL"}
    mesh_gen.create_block(nx=int(L*Lx), ny=int(L*Ly), nz=int(L*Lz),
                          x_offset=int(L*Lx) + int(L*Lbx), y_offset=0, z_offset=0,
                          boundary_names=boundary_names, renumber=False, replace_nodes=False)
    # Above Cavity
    boundary_names = {"west": None, "east": None, "south": None, "north": "TOP", "back": None, "front": None}
    mesh_gen.create_block(nx=int(L*Lbx), ny=int(L*Ly), nz=int(L*Lbz),
                          x_offset=int(L*Lx), y_offset=0, z_offset=int(L*(Lz-Lbz)/2),
                          boundary_names=boundary_names, renumber=False)
    # Front Cavity
    boundary_names = {"west": None, "east": None, "south": "BOT", "north": "TOP", "back": None, "front": "WALL"}
    mesh_gen.create_block(nx=int(L*Lbx), ny=int(L*Ly), nz=int(L*Lz) - (int(L*Lbz) + int(L*(Lz-Lbz)/2)),
                          x_offset=int(L*Lx), y_offset=0, z_offset=int(L*Lbz) + int(L*(Lz-Lbz)/2),
                          boundary_names=boundary_names, renumber=False)
    # Back Cavity
    boundary_names = {"west": None, "east": None, "south": "BOT", "north": "TOP", "back": "WALL", "front": None}
    mesh_gen.create_block(nx=int(L*Lbx), ny=int(L*Ly), nz=int(L*(Lz-Lbz)/2),
                          x_offset=int(L*Lx), y_offset=0, z_offset=0,
                          boundary_names=boundary_names, renumber=False)
    # Cavity
    boundary_names = {"west": "CAVITY", "east": "CAVITY", "south": "CAVITY",
                      "north": None, "back": "CAVITY", "front": "CAVITY"}
    mesh_gen.create_block(nx=int(L*Lbx), ny=int(L*Lby), nz=int(L*Lbz),
                          x_offset=int(L*Lx), y_offset=-int(L*Lby), z_offset=int(L*(Lz-Lbz)/2),
                          boundary_names=boundary_names, renumber=False)

    mesh_gen.renumber_all()
    return mesh_gen.mesh


if __name__ == "__main__":
    main()
