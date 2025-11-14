import random
from typing import Tuple, List
import numpy as np

def generate_fracture_data(
    Nf: int,
    domain: Tuple[float, float, float, float, float, float] = (-1000, 1000, -1000, 1000, 0, 200),
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic fracture input data.

    Parameters
    ----------
    Nf : int
        Number of fractures to generate.
    domain : tuple (xmin, xmax, ymin, ymax, zmin, zmax)
        Spatial bounding box for fracture centers.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    centers : np.ndarray
        Flat array of length 3*Nf: [x0,y0,z0, x1,y1,z1, ..., xNf-1,yNf-1,zNf-1]
    thetas : np.ndarray
        Array of length Nf: orientation angles in **degrees**, clockwise from +y axis.

    Example
    -------
    >>> centers, thetas = generate_fracture_data(100)
    >>> L, H = 200.0, 80.0
    >>> fractures = build_fractures(centers, thetas, L, H)
    """
    random.seed(seed)

    xmin, xmax, ymin, ymax, zmin, zmax = domain

    # Pre-allocate arrays instead of lists for efficiency
    centers = np.zeros(3 * Nf, dtype=np.float64)
    thetas = np.zeros(Nf, dtype=np.float64)

    for i in range(Nf):
        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)
        z = random.uniform(zmin, zmax)
        theta = random.uniform(0, 360)  # degrees, clockwise from +y

        centers[3*i] = x
        centers[3*i + 1] = y
        centers[3*i + 2] = z
        thetas[i] = theta

    return centers, thetas
