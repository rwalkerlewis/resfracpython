import numpy as np
from geometry import Point3D
from fracture import create_fracture_elements
from scipy.sparse import csr_matrix
from testing.random_fractures import generate_fracture_data
from plotting import plot_grid_with_fracture_counts

# Test with 10 fractures (original setting)
num_fractures = 10
fracture_length = 200.0
fracture_height = 80.0
centers, thetas = generate_fracture_data(num_fractures, seed=42)

centers2D_temp = centers.reshape(-1, 3)
x_min_temp = np.min(centers2D_temp[:, 0])
x_max_temp = np.max(centers2D_temp[:, 0])
y_min_temp = np.min(centers2D_temp[:, 1])
y_max_temp = np.max(centers2D_temp[:, 1])

padding = max(fracture_length, fracture_height) / 3.0
x_min_temp -= padding
x_max_temp += padding
y_min_temp -= padding
y_max_temp += padding

x_min = np.floor(x_min_temp)
x_max = np.ceil(x_max_temp)
y_min = np.floor(y_min_temp)
y_max = np.ceil(y_max_temp)

cell_size = float(max(fracture_length, fracture_height)) / 4.0
dx = dy = cell_size
grid_count_x = int(np.ceil((x_max - x_min) / dx)) if x_max > x_min else 1
grid_count_y = int(np.ceil((y_max - y_min) / dy)) if y_max > y_min else 1
grid_size = max(1, max(grid_count_x, grid_count_y))

dx = (x_max - x_min) / grid_size
dy = (y_max - y_min) / grid_size

fractures, all_corners_x_array, all_corners_y_array, intersections = create_fracture_elements(
    centers=centers, orientations=thetas, length=fracture_length, height=fracture_height,
    x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max, grid_size=grid_size
)
nf = len(fractures)

num_cells = grid_size * grid_size
frac_idx_arr, i_arr, j_arr = np.where(intersections)
cell_idx_arr = i_arr * grid_size + j_arr

if len(frac_idx_arr) > 0:
    grid_sparse = csr_matrix(
        (np.ones(len(frac_idx_arr), dtype=np.int8), (cell_idx_arr, frac_idx_arr)),
        shape=(num_cells, nf),
        dtype=np.int8
    )
else:
    grid_sparse = csr_matrix((num_cells, nf), dtype=np.int8)

# Test plotting
plot_grid_with_fracture_counts(
    grid_sparse_csr=grid_sparse,
    x_min=x_min,
    x_max=x_max,
    y_min=y_min,
    y_max=y_max,
    grid_size=grid_size,
    cell_size=cell_size,
    centers=centers,
    thetas=thetas,
    length=fracture_length,
    height=fracture_height
)
