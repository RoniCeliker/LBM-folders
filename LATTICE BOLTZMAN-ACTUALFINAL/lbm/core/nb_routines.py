import numpy as np
import numba as nb
from numba import cuda
from numba import config
config.CUDA_DEBUGINFO = 1


def density(f_device,rho_device):
    # Shape of the input array
    n = f_device.shape[1]
    # Configure grid and block sizes
    threads_per_block = 256
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

    # Launch the kernel
    density_cuda[blocks_per_grid, threads_per_block](f_device, rho_device,n)

@cuda.jit
def density_cuda(f, rho,n):
    # Get the thread ID
    j = cuda.grid(1)  # 1D grid for lattice nodes

    # Ensure the thread ID is within bounds
    if j < n:
        sum_density = 0.0
        # Sum over velocity directions (axis 0)
        for i in range(f.shape[0]):
            sum_density += f[i, j]
        rho[j] = sum_density

def velocity(f_device, rho_device, u_device, c_device, d):
    # Dimensions
    n = f_device.shape[1]         # Number of lattice nodes
    # Configure grid and block sizes
    threads_per_block = (16, d)  # Tune based on GPU
    blocks_per_grid_x = (n + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = 1
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Launch the kernel
    velocity_cuda[blocks_per_grid, threads_per_block](f_device, rho_device, u_device, c_device, d, n)

@cuda.jit
def velocity_cuda(f, rho, u, c, d, n):
    x, y = cuda.grid(2)  # Map threads to lattice nodes and velocity directions
    
    if x < n and y < d:  # Bounds check
        dot_product = 0.0
        for k in range(f.shape[0]):  # Iterate over velocity directions
            dot_product += c[y, k] * f[k, x]
        u[x, y] = dot_product / rho[x]



def force_velocity(f_device, rho_device, g_device, c_device, u_device, d): ###########
    n = f_device.shape[1]  # Number of lattice nodes
    # Configure the CUDA kernel
    threads_per_block = (16, d)  # Tune this based on GPU
    blocks_per_grid_x = (n + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = 1
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Launch the kernel
    force_velocity_cuda[blocks_per_grid, threads_per_block](
        f_device, rho_device, u_device, g_device, c_device, d, n
    )

@cuda.jit
def force_velocity_cuda(f, rho, u, g, c, d, n):
    # Get the thread indices
    x, y = cuda.grid(2)  # x: lattice node index, y: velocity direction index

    # Ensure thread indices are within bounds
    if x < n and y < d:
        # Compute dot product
        dot_product = 0.0
        for k in range(c.shape[1]):  # c.shape[1] is the number of nodes in f
            dot_product += c[y, k] * f[k, x]
        u[x, y] = dot_product

        # Add gravitational term (shared across all lattice nodes)
        u[x, y] += 0.5 * g[y]

        # Divide by density (scaling velocity components)
        u[x, y] /= rho[x]


def source_force(S_device, S_const_device, u_device, g_device, c_device, q, inv_cs_2):
    # Configure the CUDA kernel
    threads_per_block = 8
    blocks_per_grid = (q + threads_per_block - 1) // threads_per_block
    
    # Launch CUDA kernel
    source_force_cuda[blocks_per_grid, threads_per_block](
        S_device, S_const_device, u_device, c_device, q, inv_cs_2
    )

@cuda.jit
def source_force_cuda(S, S_const, u, c, q, inv_cs_2):
    i = cuda.grid(1)  # Get the index of the thread in 1D
    if i < q:
        num_elements = u.shape[0]  # Number of elements in the first dimension of `u`
        dim = c.shape[1]  # Dimensionality of `u` and `c`
        
        # shared memory test, seems very effective
        tmp = cuda.shared.array(shape=(num_elements, dim), dtype=np.float32)
        for j in range(num_elements):
            dot_product = 0.0
            for d in range(dim):
                dot_product += u[j, d] * c[i, d]
            
            for d in range(dim):
                tmp[j, d] = (dot_product * inv_cs_2 + 1) * c[i, d] - u[j, d]
        
        sum_tmp = cuda.shared.array(shape=(num_elements,), dtype=np.float32)
        for j in range(num_elements):
            sum_tmp[j] = 0.0
            for d in range(dim):
                sum_tmp[j] += tmp[j, d]
        
        for j in range(num_elements):
            S[i, j] = S_const[i] * sum_tmp[j]



def force(rho_device, force_vector_device, g_device, d):
    n = len(rho_device)         # Number of lattice nodes
    # Configure grid and block sizes
    threads_per_block = (16, d)  # Tune based on GPU
    blocks_per_grid_x = (n + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = 1
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

    # Launch the kernel
    force_cuda[blocks_per_grid, threads_per_block](g_device, rho_device, force_vector_device, d, n)

@cuda.jit
def force_cuda(g, rho, force_vector, d, n):
    # Get thread indices
    j, i = cuda.grid(2)  # j: lattice node index, i: velocity direction index

    # Ensure thread indices are within bounds
    if j < n and i < d:
        g[j, i] = rho[j] * force_vector[i]

'''def equilibrium(feq_device, u_device, rho_device, q, w_device, c_device, inv_cs_2, inv_cs_4, d):
    # Number of lattice nodes
    n = feq_device.shape[1]  # feq_device assumed shape (q, n)
    print(n)
    print(q)
    print(feq_device.shape[0])
    # Configure grid and block sizes
    threads_per_block = (16, 16)
    blocks_per_grid_x = (q + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (n + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
    if blocks_per_grid_x > 65535 or blocks_per_grid_y > 65535:
        raise ValueError("Grid dimensions exceed hardware limits!")

    # Launch CUDA kernel
    equilibrium_cuda[blocks_per_grid, threads_per_block](
        feq_device, u_device, rho_device, w_device, c_device, inv_cs_2, inv_cs_4
    )'''

'''@cuda.jit
def equilibrium_cuda(feq, u, rho, w, c, inv_cs_2, inv_cs_4):
    i, j = cuda.grid(2)  # i: velocity direction, j: lattice node

    if i < feq.shape[0] and j < feq.shape[1]:
        uu = 0.0
        for d in range(u.shape[1]):  # Velocity components
            uu += u[j, d]**2

        uc = 0.0
        for d in range(u.shape[1]):
            uc += u[j, d] * c[i, d]

        feq[i, j] = w[i] * rho[j] * (
            1 + uc * inv_cs_2 + 0.5 * (uc**2) * inv_cs_4 - 0.5 * uu * inv_cs_2
        )'''

def equilibrium(feq_device, u_device, rho_device, q, w_device, c_device, inv_cs_2, inv_cs_4, d,):
    n = feq_device.shape[1]
    threads_per_block = (16, 16)  # Limit to reasonable block size
    max_blocks_y = 65535  # Maximum blocks per grid dimension

    # Split the grid into multiple z-dimensions if needed
    blocks_per_grid_x = (q + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (n + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid_z = (blocks_per_grid_y + max_blocks_y - 1) // max_blocks_y
    blocks_per_grid_y = min(blocks_per_grid_y, max_blocks_y)  # Adjust y-dimension

    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)

    # Launch the kernel
    equilibrium_cuda[blocks_per_grid, threads_per_block](
        feq_device, u_device, rho_device, w_device, c_device, inv_cs_2, inv_cs_4
    )

@cuda.jit
def equilibrium_cuda(feq, u, rho, w, c, inv_cs_2, inv_cs_4):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x  # Velocity direction
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    z = cuda.blockIdx.z
    j = y + z * cuda.gridDim.y * cuda.blockDim.y  # Global index for lattice nodes

    if i < feq.shape[0] and j < feq.shape[1]:
        uu = 0.0
        for d in range(u.shape[1]):  # Velocity components
            uu += u[j, d]**2

        uc = 0.0
        for d in range(u.shape[1]):
            uc += u[j, d] * c[i, d]

        feq[i, j] = w[i] * rho[j] * (
            1 + uc * inv_cs_2 + 0.5 * (uc**2) * inv_cs_4 - 0.5 * uu * inv_cs_2
        )

def collision_bgk(f_device, feq_device, source_device, omega):
    # Dimensions
    q, n = f_device.shape  # f_device is assumed to have shape (q, n)

    # Configure grid and block sizes
    threads_per_block = (16, 16)  # Tune based on GPU
    max_blocks_y = 65535  # Maximum blocks per grid dimension (y)
    blocks_per_grid_x = (q + threads_per_block[0] - 1) // threads_per_block[0]
    total_blocks_y = (n + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid_y = min(total_blocks_y, max_blocks_y)
    '''    blocks_per_grid_y = (n + threads_per_block[1] - 1) // threads_per_block[1]'''
    '''blocks_per_grid_z = min(blocks_per_grid_y, max_blocks_y)'''
    blocks_per_grid_z = (total_blocks_y + max_blocks_y - 1) // max_blocks_y
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z)


    # Launch the CUDA kernel
    collision_bgk_cuda[blocks_per_grid, threads_per_block](
        f_device, feq_device, source_device, omega, q, n
    )

@cuda.jit
def collision_bgk_cuda(f, feq, source, omega, q, n):
    # Map threads to 3D grid indices
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x  # Velocity direction
    y = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    z = cuda.blockIdx.z
    j = y + z * cuda.gridDim.y * cuda.blockDim.y  # Lattice node index

    # Ensure thread indices are within bounds
    if i < q and j < n:
        f[i, j] += source[i, j] - omega * (f[i, j] - feq[i, j])

'''@cuda.jit
def collision_bgk_cuda(f, feq, source, omega,q, n):
    i, j = cuda.grid(2)  # Map threads to 2D indices

    if i < n and j < q:  # Ensure thread indices are within bounds
        f[i, j] += source[i, j] - omega * (f[i, j] - feq[i, j])'''

#def collision_trt(f_device, feq_device, source, i_opp, omega_plus, omega_minus):

    # Configure grid and block sizes
"""     threads_per_block = (16, 16)  # Tune based on GPU
    blocks_per_grid_x = (f.shape[0] + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_per_grid_y = (f.shape[1] + threads_per_block[1] - 1) // threads_per_block[1]
    blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y) """

    # Launch the CUDA kernel
"""     collision_trt_cuda[blocks_per_grid, threads_per_block](
        f_device, feq_device, source_device, i_opp_device, omega_plus, omega_minus
    )
 """

@cuda.jit
def collision_trt_cuda(f, feq, source, i_opp, omega_plus, omega_minus):
    i, j = cuda.grid(2)  # Map threads to velocity directions and lattice nodes

    if i < f.shape[0] and j < f.shape[1]:  # Ensure thread indices are within bounds
        # Compute f and feq differences
        f_diff = f[i, j] - feq[i, j]
        ''' opp_diff = f[i_opp[i], j] - feq[i_opp[i], j]'''

        # Update f using TRT collision rules
        f[i, j] += source[i, j] - 0.5 * (omega_plus + omega_minus) * f_diff
        f[i_opp[i], j] -= 0.5 * (omega_plus - omega_minus) * f_diff


def streaming_bulk(f_device, f0_device, q, bulk_streaming_map_device, bulk_cell_index_device):
    # Configure grid and block sizes
    n_cells = bulk_cell_index_device.size
    threads_per_block = 256  # Tune based on GPU
    blocks_per_grid = (n_cells + threads_per_block - 1) // threads_per_block


    # Launch the CUDA kernel
    streaming_bulk_cuda[blocks_per_grid, threads_per_block](
        f_device, f0_device, bulk_streaming_map_device, bulk_cell_index_device, q, n_cells
    )


@cuda.jit
def streaming_bulk_cuda(f, f0, streaming_map_bulk, bulk_cell_index, q, n_cells):
    # Compute the global thread ID
    tid = cuda.grid(1)
    if tid < n_cells:
        # Get the bulk cell index for this thread
        cell_id = bulk_cell_index[np.uint64(tid)] #hotfix, bottleneck?
        # Iterate over the velocity set (q)
        for i in range(q):
            f[i, cell_id] = f0[i, streaming_map_bulk[i, tid]]



def streaming_boundary(f_device, f0_device, streaming_map_boundary_device):
    # Configure grid and block sizes
    threads_per_block = 256  # Tune based on GPU
    blocks_per_grid = (streaming_map_boundary_device.shape[0] + threads_per_block - 1) // threads_per_block

    # Launch the CUDA kernel
    streaming_boundary_cuda[blocks_per_grid, threads_per_block](
        f_device, f0_device, streaming_map_boundary_device
    )

@cuda.jit
def streaming_boundary_cuda(f, f0, streaming_map_boundary):
    idx = cuda.grid(1)  # Map each thread to one boundary operation

    if idx < streaming_map_boundary.shape[0]:  # Ensure thread index is within bounds
        #bottlneck?
        cell_idx = streaming_map_boundary[idx, 0]
        cell0_idx = streaming_map_boundary[idx, 1]
        i = streaming_map_boundary[idx, 2]
        f[i, cell_idx] = f0[i, cell0_idx]


def boundary_bounce_back(f_device, f0_device, streaming_map_device, wall_data_device):
    # Number of lattice nodes
    n = streaming_map_device.shape[0]

    # Configure grid and block sizes
    threads_per_block = 256  # Tune based on GPU
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

    # Launch the CUDA kernel
    boundary_bounce_back_cuda[blocks_per_grid, threads_per_block](
        f_device, f0_device, streaming_map_device, wall_data_device
    )

@cuda.jit
def boundary_bounce_back_cuda(f, f0, streaming_map, wall_data):
    idx = cuda.grid(1)  # Get the index of the thread in 1D

    if idx < streaming_map.shape[0]:  # Ensure thread index is within bounds
        cell_index = streaming_map[idx, 0]
        i_opp = streaming_map[idx, 1]
        i = streaming_map[idx, 2]
        f[i_opp, cell_index] = f0[i, cell_index] + wall_data[i]

""" @nb.njit
def wall_data_anti_bounce_back(u, c, w, inv_cs_2, inv_cs_4, rho_wall, i_c, i_c_n, ii):
    u_wall = 1.5*u[i_c] - 0.5*u[i_c_n]
    #print(f"\t\t\t\t u_w =  1.5*u[{i_c}] - 0.5*u[{i_c_n}]")
    return 2.0 * w[ii] * rho_wall * \
             (1 +
              0.5 * np.dot(c[ii], u_wall)**2 * inv_cs_4 -
              0.5 * np.dot(u_wall, u_wall) * inv_cs_2) """

""" def wall_data_anti_bounce_back_host(u_device, c_device, w_device, inv_cs_2, inv_cs_4, rho_wall,
                                    i_c_device, i_c_n_device, ii_device, result_device):
    # Get the number of elements to process
    n = ii_device.size

    # Configure threads and blocks
    threads_per_block = 256  # Optimize based on your GPU
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

    # Launch the kernel
    wall_data_anti_bounce_back_kernel[blocks_per_grid, threads_per_block](
        u_device, c_device, w_device, inv_cs_2, inv_cs_4, rho_wall,
        i_c_device, i_c_n_device, ii_device, result_device
    )

    # Synchronize to ensure the kernel execution is complete
    cuda.synchronize()
 """

""" @cuda.jit
def wall_data_anti_bounce_back_kernel(u, c, w, inv_cs_2, inv_cs_4, rho_wall, i_c, i_c_n, ii, result):
    # Compute the global thread ID
    tid = cuda.grid(1)

    # Check bounds
    if tid < ii.size:  # Ensure thread index is within bounds of ii
        # Calculate u_wall for the current thread
        u_wall = 1.5 * u[i_c[tid]] - 0.5 * u[i_c_n[tid]]

        # Compute the anti-bounce-back value
        dot_c_u = c[ii[tid]].dot(u_wall)
        dot_u_u = u_wall.dot(u_wall)
        result[tid] = 2.0 * w[ii[tid]] * rho_wall * (
            1 +
            0.5 * dot_c_u**2 * inv_cs_4 -
            0.5 * dot_u_u * inv_cs_2
        )
 """

""" @nb.njit(parallel=True)
def boundary_anti_bounce_backooga(f, f0, u, streaming_map, rho_wall, w, c, inv_cs_2, inv_cs_4):
    # streaming map contains:
    # [i, 0]: cell index
    # [i, 1]: cell neighbour index
    # [i, 2]: population index destination
    # [i, 3]: population index source
    for idx in nb.prange(streaming_map.shape[0]):
        i_c = streaming_map[idx, 0]
        i_c_n = streaming_map[idx, 1]
        i_opp = streaming_map[idx, 2]
        i = streaming_map[idx, 3]
        #print(f"boundary_anti_bounce_back \tf{i_opp}:{i_c} = - f{i}:{i_c}")
        f[i_opp, i_c] = - f0[i, i_c] + wall_data_anti_bounce_back(u, c, w, inv_cs_2, inv_cs_4,
                                                                  rho_wall, i_c, i_c_n, i) """


def boundary_anti_bounce_back(f_device, f0_device, u_device, streaming_map_device,
                                   density, w_device, c_device, inv_cs_2, inv_cs_4):
    # Determine the number of rows in the streaming_map
    n = streaming_map_device.shape[0]

    # Configure threads and blocks
    threads_per_block = 256  # Optimal value depends on your GPU
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
    # Launch the kernel
    boundary_anti_bounce_back_kernel[blocks_per_grid, threads_per_block](
        f_device, f0_device, u_device, streaming_map_device,
        density, w_device, c_device, inv_cs_2, inv_cs_4
    )
    cuda.synchronize()

@cuda.jit
def boundary_anti_bounce_back_kernel(f, f0, u, streaming_map, density, w, c, inv_cs_2, inv_cs_4):
    tid = cuda.grid(1)  # Global thread ID

    if tid < streaming_map.shape[0]:  # Check bounds
        i_c = streaming_map[tid, 0]
        i_c_n = streaming_map[tid, 1]
        i_opp = streaming_map[tid, 2]
        i = streaming_map[tid, 3]



        # Inline wall_data_anti_bounce_back logic
        u_wall = cuda.local.array(2, dtype=np.float32)  # Local array for u_wall
        for j in range(2):  # Explicit element-wise operations
            u_wall[j] = np.float32(1.5) * u[i_c, j] - np.float32(0.5) * u[i_c_n, j]

        dot_c_u = c[i, 0] * u_wall[0] + c[i, 1] * u_wall[1]  # Dot product of c[i] and u_wall
        dot_u_u = u_wall[0]**2 + u_wall[1]**2  # Dot product of u_wall with itself

        wall_value = np.float32(2.0) * w[i] * density * (
            np.float32(1) + np.float32(0.5) * dot_c_u**2 * inv_cs_4 - np.float32(0.5) * dot_u_u * inv_cs_2
        )

        f[i_opp, i_c] = -f0[i, i_c] + wall_value

""" @nb.njit(parallel=True)
def boundary_symmetryyy(f, f0, streaming_map):
    # streaming map contains:
    # [i, 0]: cell index
    # [i, 1]: cell symmetry index
    # [i, 2]: population index destination
    # [i, 3]: population index source
    for idx in nb.prange(streaming_map.shape[0]):
        i_c = streaming_map[idx, 0]
        i_c_s = streaming_map[idx, 1]
        i_s = streaming_map[idx, 2]
        i = streaming_map[idx, 3]
        #print(f"boundary_symmetry \tf{i_s}:{i_c} = f{i}:{i_c_s}")
        f[i_s, i_c] = f0[i, i_c_s] """

def boundary_symmetry(f_device, f0_device, streaming_map_device):
    # Determine the number of rows in the streaming_map
    n = streaming_map_device.shape[0]

    # Configure threads and blocks
    threads_per_block = 256  # Optimal value depends on your GPU
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

    # Launch the kernel
    boundary_symmetry_kernel[blocks_per_grid, threads_per_block](f_device, f0_device, streaming_map_device)

    # Synchronize to ensure kernel execution is complete
    cuda.synchronize()

@cuda.jit
def boundary_symmetry_kernel(f, f0, streaming_map):
    # Compute the global thread ID
    tid = cuda.grid(1)

    # Ensure thread index is within bounds
    if tid < streaming_map.shape[0]:
        # Extract streaming map values
        i_c = streaming_map[tid, 0]  # Cell index
        i_c_s = streaming_map[tid, 1]  # Symmetry cell index
        i_s = streaming_map[tid, 2]  # Population index destination
        i = streaming_map[tid, 3]  # Population index source

        # Apply symmetry boundary condition
        f[i_s, i_c] = f0[i, i_c_s]

def boundary_periodic():
    pass
