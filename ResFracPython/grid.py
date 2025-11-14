"""
Grid generation and initialization module.

Handles creation of spatial grids for fracture analysis, including bounds computation,
cell size calculation, and sparse matrix construction for fracture-cell intersections.
"""

import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple
from .fracture import FractureElement, create_fracture_elements


def compute_grid_bounds(
    centers: np.ndarray,
    length: float,
    height: float,
    padding_factor: float = 1.0
) -> Tuple[float, float, float, float]:
    """
    Compute grid bounds from fracture centers with padding.
    
    Parameters
    ----------
    centers : np.ndarray
        Flat array: [x0,y0,z0, x1,y1,z1, ...]
    length : float
        Fracture length for padding calculation.
    height : float
        Fracture height for padding calculation.
    padding_factor : float, optional
        Padding as multiple of max(length, height). Default is 1.0.
    
    Returns
    -------
    tuple
        (x_min, x_max, y_min, y_max) - bounds of the domain
    """
    centers2D = centers.reshape(-1, 3)
    x_min_temp = np.min(centers2D[:, 0])
    x_max_temp = np.max(centers2D[:, 0])
    y_min_temp = np.min(centers2D[:, 1])
    y_max_temp = np.max(centers2D[:, 1])
    
    # Add padding to ensure fractures near edges are included
    padding = max(length, height) * padding_factor
    x_min_temp -= padding
    x_max_temp += padding
    y_min_temp -= padding
    y_max_temp += padding
    
    # Round to integer bounds
    x_min = np.floor(x_min_temp)
    x_max = np.ceil(x_max_temp)
    y_min = np.floor(y_min_temp)
    y_max = np.ceil(y_max_temp)
    
    return float(x_min), float(x_max), float(y_min), float(y_max)


def compute_grid_size(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    length: float,
    height: float,
    cell_size_factor: float = 0.25,
) -> Tuple[int, float, float]:
    """
    Compute grid size and cell dimensions.
    
    Parameters
    ----------
    x_min, x_max : float
        X-axis bounds.
    y_min, y_max : float
        Y-axis bounds.
    length : float
        Fracture length (used to determine base cell size).
    height : float
        Fracture height (used to determine base cell size).
    cell_size_factor : float, optional
        Factor to multiply the base cell size. Default is 0.25.
    
    Returns
    -------
    tuple
        (grid_size, dx, dy) - number of cells per side and cell dimensions
    """
    # Fix cell size based on fracture dimensions (1/4 of largest dimension)
    cell_size = float(max(length, height)) * cell_size_factor
    dx = dy = cell_size
    
    # Compute number of cells needed in each direction
    grid_count_x = int(np.ceil((x_max - x_min) / dx)) if x_max > x_min else 1
    grid_count_y = int(np.ceil((y_max - y_min) / dy)) if y_max > y_min else 1
    
    # Use square grid sized to cover both dimensions
    grid_size = max(1, max(grid_count_x, grid_count_y))
    
    # Recompute dx and dy to evenly divide the domain
    dx = (x_max - x_min) / grid_size
    dy = (y_max - y_min) / grid_size
    
    return grid_size, dx, dy


def create_grid(
    centers: np.ndarray,
    thetas: np.ndarray,
    length: float,
    height: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    grid_size: int
) -> Tuple[csr_matrix, int, np.ndarray]:
    """
    Create a spatial grid and compute fracture-cell intersections.
    
    Parameters
    ----------
    centers : np.ndarray
        Flat array: [x0,y0,z0, x1,y1,z1, ...]
    thetas : np.ndarray
        Orientation angles in degrees (clockwise from +y).
    length : float
        Fracture length.
    height : float
        Fracture height.
    x_min, x_max : float
        X-axis bounds.
    y_min, y_max : float
        Y-axis bounds.
    grid_size : int
        Number of cells per side.
    
    Returns
    -------
    tuple
        (grid_sparse_csr, num_fractures, fractures)
        - grid_sparse_csr: Sparse matrix of shape (num_cells, num_fractures)
        - num_fractures: Number of fractures
        - fractures: NumPy array (dtype=object) of FractureElement objects
    """
    # Create fracture elements and compute intersections
    fractures, _, _, intersections = create_fracture_elements(
        centers=centers,
        orientations=thetas,
        length=length,
        height=height,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        grid_size=grid_size
    )
    
    nf = len(fractures)
    num_cells = grid_size * grid_size
    
    # Extract indices from intersection array
    frac_idx_arr, i_arr, j_arr = np.where(intersections)
    cell_idx_arr = i_arr * grid_size + j_arr
    
    # Create sparse matrix
    if len(frac_idx_arr) > 0:
        grid_sparse = csr_matrix(
            (np.ones(len(frac_idx_arr), dtype=np.int8), (cell_idx_arr, frac_idx_arr)),
            shape=(num_cells, nf),
            dtype=np.int8
        )
    else:
        grid_sparse = csr_matrix((num_cells, nf), dtype=np.int8)
    
    return grid_sparse, nf, fractures


def initialize_grid(
    centers: np.ndarray,
    thetas: np.ndarray,
    length: float,
    height: float,
    padding_factor: float = 1.0,
    cell_size_factor: float = 0.25
) -> Tuple[csr_matrix, int, np.ndarray, float, float, float, float, int]:
    """
    Complete grid initialization: compute bounds, grid size, and create grid.
    
    This is the main entry point for grid creation.
    
    Parameters
    ----------
    centers : np.ndarray
        Flat array: [x0,y0,z0, x1,y1,z1, ...]
    thetas : np.ndarray
        Orientation angles in degrees (clockwise from +y).
    length : float
        Fracture length.
    height : float
        Fracture height.
    padding_factor : float, optional
        Padding as multiple of max(length, height). Default is 1.0.
    cell_size_factor : float, optional
        Factor to multiply the base cell size. Default is 0.25.
    
    Returns
    -------
    tuple
        (grid_sparse_csr, num_fractures, fractures, x_min, x_max, y_min, y_max, grid_size)
    """
    # Compute bounds
    x_min, x_max, y_min, y_max = compute_grid_bounds(
        centers, length, height, padding_factor
    )
    
    # Compute grid size and cell dimensions
    grid_size, dx, dy = compute_grid_size(
        x_min, x_max, y_min, y_max, length, height, cell_size_factor=0.25
    )
    
    # Create grid
    grid_sparse_csr, nf, fractures = create_grid(
        centers, thetas, length, height,
        x_min, x_max, y_min, y_max, grid_size
    )
    
    return grid_sparse_csr, nf, fractures, x_min, x_max, y_min, y_max, grid_size
