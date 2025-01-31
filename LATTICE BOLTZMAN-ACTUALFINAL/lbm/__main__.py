import os
import sys
import time
import numpy as np
from lbm.app.app import app_factory
from lbm.util.export import Export


def main():
    if len(sys.argv) < 3:
        print(f'Input: app_name output_dir')
        exit()
    app_name = str(sys.argv[1])
    output_dir = os.path.abspath(str(sys.argv[2]))
    if not os.path.isdir(output_dir):
        print(f'Creating directory: {output_dir}')
        os.makedirs(output_dir)
    if app_name not in app_factory.keys:
        print(f'{app_name} is not an existing application!')
        exit()

    print('Initializing app...')
    app = app_factory.create(app_name)
    lattice = app.init_lattice()
    lattice.init_f()
    print('DONE')
    print(f'Tau = {lattice.tau}')
    print(f'Number of cells: {lattice.mesh.n}')
    max_iterations = np.uint64(40001)
    i_mod = 100
    export = Export(lattice)
    for i in range(max_iterations):
        lattice.time_step()
        if np.mod(i, i_mod) == 0:
            total_mass = lattice.total_mass()
            print(f"iter: {i:<6}, Mass check: {total_mass}")
            lattice.display_timer()
            lattice.reset_timer()
            print("Writing to file:")
            write_cell_info = i == 0
            t0 = time.perf_counter()
            export.write_vtk(f"{output_dir}/{app_name}-{i}.vtk", write_cell_info=write_cell_info)
            t1 = time.perf_counter() - t0
            print(f'Time exporting: {t1}')
            print("DONE")
            if np.isnan(total_mass):
                print(f"Solution diverged, aborting")
                break


if __name__ == "__main__":
    main()
