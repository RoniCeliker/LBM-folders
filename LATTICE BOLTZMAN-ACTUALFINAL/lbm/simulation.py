import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import numpy as np
from lbm.util.export import Export
from lbm.core.lattice import Lattice


class Simulation:
    def __init__(self, lattice: Lattice, output_directory: str, application_name: str = "simulation"):
        self._lattice = lattice
        self._output_directory = output_directory
        self._application_name = application_name
        self._export = Export(lattice)
        self._timer = {"boundaries": 0.0, "streaming": 0.0, "macroscopic": 0.0,
                       "equilibrium": 0.0, "source_force": 0.0, "collision": 0.0}
        self._number_of_cells = np.uint32(self._lattice.mesh.n)
        self._total_time = 0.0

    def run(self, max_it: int, mod_it: int):
        os.makedirs(self._output_directory, exist_ok=True)
        self._lattice.init_f()
        for it in range(max_it):
            self._time_step()
            if np.mod(it, mod_it) == 0:
                status = self._save_it(it)
                if status != 0:
                    print(f"Solution diverged, aborting")
                    return
        # Lets also save the final iteration
        if np.mod(max_it - 1, mod_it) != 0:
            self._save_it(max_it)

    def _save_it(self, it: int) -> int:
        total_mass = self._lattice.total_mass()
        print("*************************************************")
        print(f"iter: {it:>6}, Mass check: {total_mass}")
        self._display_time()
        self._reset_time()
        print(f"MLUPs: {self._mlups(it)}")
        print(f"Total time: {self._total_time}")
        '''self._export_vtk(f"{self._output_directory}/{self._application_name}-{it}.vtk")'''
        if np.isnan(total_mass):
            return 1
        else:
            return 0

    def _mlups(self, iteration):
        """
        Millions of lattice updates per second
        """
        return (self._number_of_cells * iteration * 1e-6) / self._total_time

    def _reset_time(self):
        for key in self._timer:
            self._timer[key] = 0.0

    def _display_time(self):
        time_sum = 0.0
        for key in self._timer:
            time_sum += self._timer[key]
            print(f"\tTime {key:<16} {self._timer[key]:12.10f}")
        print(f"\tTime {'sum':<16} {time_sum:12.10f}")

    '''def _export_vtk(self, filename):
        t0 = time.perf_counter()
        self._export.write_vtk(filename)
        t1 = time.perf_counter() - t0
        print(f'Time exporting: {t1}')
    '''
    def _time_step(self):
        t0 = time.perf_counter()

        t1 = t0
        self._lattice.boundaries()
        self._timer['boundaries'] += time.perf_counter() - t1

        t1 = time.perf_counter()
        self._lattice.streaming()
        self._timer['streaming'] += time.perf_counter() - t1

        t1 = time.perf_counter()
        self._lattice.macroscopic()
        self._timer['macroscopic'] += time.perf_counter() - t1

        t1 = time.perf_counter()
        self._lattice.equilibrium()
        self._timer['equilibrium'] += time.perf_counter() - t1

        t1 = time.perf_counter()
        self._lattice.source_force()
        self._timer['source_force'] += time.perf_counter() - t1

        t1 = time.perf_counter()
        self._lattice.collision()
        self._timer['collision'] += time.perf_counter() - t1

        self._lattice.swap_f()
        self._total_time += time.perf_counter() - t0
