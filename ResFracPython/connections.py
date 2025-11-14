"""
Fracture connection detection module.

Handles identification of fracture intersections and connection mapping
using vectorized operations for efficiency.
"""

import numpy as np
from typing import List, Tuple, Set, Dict
from scipy.sparse import csr_matrix, lil_matrix
from .fracture import FractureElement
from .intersection import z_range_overlap
from .geometry import Point3D


def _check_fracture_pairs(
    frac_set1: np.ndarray,
    frac_set2: np.ndarray,
    fractures: np.ndarray,
    connections_matrix: lil_matrix,
    intersection_points: Dict[Tuple[int, int], Point3D],
    checked_pairs: Set[Tuple[int, int]]
) -> None:
    """
    Check all pairs between two sets of fractures for intersections.
    
    Helper function that consolidates the pair generation and intersection
    checking logic. Updates connections_matrix and intersection_points in-place.
    
    Parameters
    ----------
    frac_set1, frac_set2 : np.ndarray
        Arrays of fracture indices to check (can be the same array).
    fractures : List[FractureElement]
        List of FractureElement objects.
    connections_matrix : lil_matrix
        Sparse LIL matrix where entry (i,j)=1 if fractures i and j intersect.
    intersection_points : np.ndarray
        Array of Point3D or None values indexed by flattened (i,j) pair index.
    checked_pairs : Set
        Set of already-checked pairs (modified in-place).
    """
    if len(frac_set1) == 0 or len(frac_set2) == 0:
        return
    
    # Create all pairs using meshgrid
    i_indices, j_indices = np.meshgrid(
        frac_set1, frac_set2, indexing='ij'
    )
    # Flatten the arrays
    pairs_i = i_indices.flatten()
    pairs_j = j_indices.flatten()
    
    # Only keep pairs where i < j (avoid duplicates and self-pairs)
    mask = pairs_i < pairs_j
    pairs_i = pairs_i[mask]
    pairs_j = pairs_j[mask]
    
    # Check intersections for all pairs
    for i, j in zip(pairs_i, pairs_j):
        pair = (int(i), int(j))
        if pair not in checked_pairs:
            checked_pairs.add(pair)
            # Screen by z-axis overlap first (fast check)
            if z_range_overlap(fractures[i], fractures[j]):
                # If z ranges overlap, do the full intersection test
                if fractures[i].intersects(fractures[j]):
                    intersection_pt = fractures[i].get_intersection_point(fractures[j])
                    # Store connection in sparse matrix
                    connections_matrix[i, j] = 1
                    connections_matrix[j, i] = 1
                    # Store intersection point
                    if intersection_pt:
                        intersection_points[pair] = intersection_pt


def find_fracture_connections(
    grid_sparse_csr: csr_matrix,
    fractures: np.ndarray,
    grid_size: int
) -> Tuple[csr_matrix, Dict[Tuple[int, int], Point3D]]:
    """
    Find all fracture connections (intersections) using vectorized operations.
    
    This function checks for intersections between fractures in the same cell
    and in neighboring cells without explicit nested loops. Also computes the
    3D intersection point for each pair of connected fractures.
    
    Parameters
    ----------
    grid_sparse_csr : csr_matrix
        Sparse matrix of shape (num_cells, num_fractures) where entry (i, j) 
        indicates whether fracture j intersects grid cell i.
    fractures : np.ndarray
        NumPy array (dtype=object) of FractureElement objects.
    grid_size : int
        Number of cells per side (grid_size x grid_size).
    
    Returns
    -------
    Tuple[csr_matrix, Dict]
        - connections: Sparse CSR matrix where entry (i,j)=1 if fractures i and j 
          are connected. Matrix is symmetric (connections[i,j] = connections[j,i]).
        - intersection_points: Dictionary with keys (i, j) tuples and Point3D values 
          for intersection coordinates.
    """
    nf = len(fractures)
    # Use LIL matrix for efficient incremental construction
    connections_matrix = lil_matrix((nf, nf), dtype=np.int32)
    intersection_points: Dict[Tuple[int, int], Point3D] = {}
    checked_pairs: Set[Tuple[int, int]] = set()
    
    # Find non-empty cells
    cell_row_sums = np.asarray(grid_sparse_csr.sum(axis=1)).flatten()
    non_empty_cells = np.where(cell_row_sums > 0)[0]
    
    # Pre-compute neighbor offsets (8 surrounding cells)
    neighbor_offsets = np.array([
        [-1,-1], [-1,0], [-1,1],
        [0, -1],         [0, 1],
        [1, -1], [1, 0], [1, 1]
    ])
    
    # Process each non-empty cell
    for cell_idx in non_empty_cells:
        # Get fractures in this cell
        fracture_indices = grid_sparse_csr[cell_idx, :].nonzero()[1]
        
        if len(fracture_indices) == 0:
            continue
        
        # --- Check pairs within this cell ---
        _check_fracture_pairs(
            fracture_indices, fracture_indices,
            fractures, connections_matrix, intersection_points, checked_pairs
        )
        
        # --- Check with fractures in neighboring cells ---
        ix = cell_idx // grid_size
        iy = cell_idx % grid_size
        
        # Vectorized neighbor coordinate computation
        neighbor_ix = neighbor_offsets[:, 0] + ix
        neighbor_iy = neighbor_offsets[:, 1] + iy
        
        # Filter valid neighbors (within bounds)
        valid_mask = (neighbor_ix >= 0) & (neighbor_ix < grid_size) & \
                     (neighbor_iy >= 0) & (neighbor_iy < grid_size)
        neighbor_ix_valid = neighbor_ix[valid_mask]
        neighbor_iy_valid = neighbor_iy[valid_mask]
        neighbor_cell_indices = neighbor_ix_valid * grid_size + neighbor_iy_valid
        
        # For each neighboring cell, find intersections with current cell fractures
        for neighbor_cell_idx in neighbor_cell_indices:
            neighbor_fractures = grid_sparse_csr[neighbor_cell_idx, :].nonzero()[1]
            
            # Check all pairs between current cell and neighbor cell fractures
            _check_fracture_pairs(
                fracture_indices, neighbor_fractures,
                fractures, connections_matrix, intersection_points, checked_pairs
            )
    
    # Convert LIL to CSR format for efficient operations
    connections_csr = connections_matrix.tocsr()
    
    return connections_csr, intersection_points
