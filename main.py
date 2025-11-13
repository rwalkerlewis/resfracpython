import numpy as np
import matplotlib.pyplot as plt
import random_fractures
from typing import List, Tuple

from geometry import Point3D
from fracture import FractureElement
from scipy.sparse import dok_matrix, csr_matrix


def create_fracture_elements(
    centers: List[float],
    orientations: List[float],
    length: float,
    height: float
) -> List[FractureElement]:
    """
    Create a list of FractureElement objects from input vectors.
    
    Args:
        centers: Flat list of length Nf*3: [x0, y0, z0, x1, y1, z1, ...]
        orientations: List of length Nf: [theta0, theta1, ...]
        length: Length L of each fracture
        height: Height H of each fracture
    
    Returns:
        List of FractureElement objects
    """
    nf = len(orientations)
    if len(centers) != nf * 3:
        raise ValueError(f"centers length ({len(centers)}) must be {nf} * 3 = {nf * 3}")
    
    fractures = []
    for i in range(nf):
        x = centers[i * 3]
        y = centers[i * 3 + 1]
        z = centers[i * 3 + 2]
        center = Point3D(x, y, z)
        orientation = orientations[i]
        fracture = FractureElement(center, length, height, orientation)
        fractures.append(fracture)
        
    return fractures

centers, thetas = random_fractures.generate_fracture_data(
        Nf=30,
        domain=(-800, 800, -800, 800, 0, 100),
        seed=42
    )

L, H = 200.0, 80.0

    # Plot top-down view
random_fractures.plot_fractures_topdown(centers, thetas, L, H, title="Random Fractures (Top View)")

fractures = create_fracture_elements(centers=centers, orientations=thetas, length=L, height=H)
nf = len(fractures)

# Use spatial grid

