"""
Small testing helpers that produce deterministic fracture datasets for unit tests.
"""

import numpy as np
from typing import Tuple


def crossing_fractures() -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Return a simple test dataset with four fractures arranged to intersect.

    Returns
    -------
    centers : np.ndarray
        Flat array of length 12: [x0,y0,z0, x1,y1,z1, x2,y2,z2, x3,y3,z3]
    orientations : np.ndarray
        Array of length 4 with orientations in degrees (clockwise from +y)
    length : float
        Fracture length
    height : float
        Fracture height

    Description of configuration
    ----------------------------
    Fractures are placed at the corners of a square (top-down):
    F0: center (-5, -5, 0)
    F1: center ( 5, -5, 0)
    F2: center ( 5,  5, 0)
    F3: center (-5,  5, 0)
    None of the centers are at (0,0).
    """
    centers = np.array([
        -5.0,  0.0,  0.0,   # F0 left side, vertical
         5.0,  0.0,  0.0,   # F1 right side, vertical
         0.0, -5.0,  0.0,   # F2 bottom side, horizontal
         0.0,  5.0,  0.0    # F3 top side, horizontal
    ], dtype=float)

    # Keep orientations so fractures span across the square edges;
    # here we'll use vertical for the left/right fractures and horizontal
    # for the top/bottom fractures.
    orientations = np.array([
        0.0,  # F0 vertical (left)
        0.0,  # F1 vertical (right)
        90.0,   # F2 horizontal (top)
        90.0    # F3 horizontal (bottom)
    ], dtype=float)

    length = 12.0
    height = 6.0

    return centers, orientations, length, height


def vertical_separation_fractures() -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Return a test dataset with fractures at significantly different depths.
    
    This simulates a realistic scenario where fractures exist at different
    vertical (z) levels in the subsurface, which would not intersect
    even if their xy projections overlap.

    Returns
    -------
    centers : np.ndarray
        Flat array with fracture centers at varying depths
    orientations : np.ndarray
        Array of orientations in degrees (clockwise from +y)
    length : float
        Fracture length
    height : float
        Fracture height

    Description of configuration
    ----------------------------
    Creates 6 fractures with two sets of 3 fractures each:
    
    Shallow set (z ≈ 0):
    F0: center (0, 0, 0)     - horizontal
    F1: center (0, 0, 5)     - vertical
    F2: center (5, 5, 10)    - diagonal
    
    Deep set (z ≈ 100+):
    F3: center (0, 0, 100)   - horizontal (directly below F0)
    F4: center (0, 0, 110)   - vertical (directly below F1)
    F5: center (5, 5, 120)   - diagonal (directly below F2)
    
    The deep fractures have the same xy projections as the shallow ones
    but are separated by large vertical distances, so they should not
    intersect despite xy overlap.
    """
    centers = np.array([
        # Shallow set
        0.0,   0.0,    4.0,    # F0 - shallow horizontal
        0.0,   0.0,   15.0,    # F1 - shallow vertical
        5.0,   5.0,   25.0,    # F2 - shallow diagonal
        
        # Deep set (same xy, different z)
        0.0,   0.0,  100.0,    # F3 - deep horizontal (below F0)
        0.0,   0.0,  110.0,    # F4 - deep vertical (below F1)
        5.0,   5.0,  120.0,    # F5 - deep diagonal (below F2)
    ], dtype=float)
    
    orientations = np.array([
        0.0,    # F0 - vertical strike
        90.0,   # F1 - horizontal strike
        45.0,   # F2 - diagonal strike
        0.0,    # F3 - vertical strike
        90.0,   # F4 - horizontal strike
        45.0,   # F5 - diagonal strike
    ], dtype=float)
    
    length = 20.0
    height = 8.0
    
    return centers, orientations, length, height


def intersecting_vertical_fractures() -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Return a test dataset with fractures at different depths that DO intersect.
    
    Unlike vertical_separation_fractures, this creates fractures that actually
    cross in 3D space despite being at different depths.

    Returns
    -------
    centers : np.ndarray
        Flat array with fracture centers
    orientations : np.ndarray
        Array of orientations in degrees (clockwise from +y)
    length : float
        Fracture length
    height : float
        Fracture height

    Description of configuration
    ----------------------------
    Creates 4 fractures that form an X pattern when viewed from above,
    but at different depths:
    
    F0: center (0, 0, 0)     - vertical orientation, passes through origin
    F1: center (0, 0, 10)    - vertical orientation, offset in z
    F2: center (5, 0, 5)     - horizontal orientation, crosses F0 and F1
    F3: center (-5, 0, 5)    - horizontal orientation, crosses F0 and F1
    
    Expected intersections:
    - F0 should intersect with F2 and F3 (crosses at different z)
    - F1 should intersect with F2 and F3 (crosses at different z)
    - F2 and F3 may or may not intersect depending on z overlap
    """
    centers = np.array([
        0.0,   0.0,   0.0,     # F0 - vertical fracture at origin
        0.0,   0.0,  10.0,     # F1 - vertical fracture offset in z
        5.0,   0.0,   5.0,     # F2 - horizontal fracture
       -5.0,   0.0,   5.0,     # F3 - horizontal fracture on opposite side
    ], dtype=float)
    
    orientations = np.array([
        0.0,    # F0 - vertical orientation
        0.0,    # F1 - vertical orientation
        90.0,   # F2 - horizontal orientation
        90.0,   # F3 - horizontal orientation
    ], dtype=float)
    
    length = 20.0
    height = 15.0
    
    return centers, orientations, length, height
