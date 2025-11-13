import random
from typing import Tuple, List
import matplotlib.pyplot as plt
import numpy as np

def generate_fracture_data(
    Nf: int,
    domain: Tuple[float, float, float, float, float, float] = (-1000, 1000, -1000, 1000, 0, 200),
    seed: int = 42
) -> Tuple[List[float], List[float]]:
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
    centers : List[float]
        Flat list of length 3*Nf: [x0,y0,z0, x1,y1,z1, ..., xNf-1,yNf-1,zNf-1]
    thetas : List[float]
        List of length Nf: orientation angles in **degrees**, clockwise from +y axis.

    Example
    -------
    >>> centers, thetas = generate_fracture_data(100)
    >>> L, H = 200.0, 80.0
    >>> fractures = build_fractures(centers, thetas, L, H)
    """
    random.seed(seed)

    xmin, xmax, ymin, ymax, zmin, zmax = domain

    centers: List[float] = []
    thetas: List[float] = []

    for _ in range(Nf):
        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)
        z = random.uniform(zmin, zmax)
        theta = random.uniform(0, 360)  # degrees, clockwise from +y

        centers.extend([x, y, z])
        thetas.append(theta)

    return centers, thetas

def plot_fractures_topdown(
    centers: List[float],
    thetas: List[float],
    L: float,
    H: float,
    title: str = "Fractures (Top-Down View)",
    figsize: Tuple[float, float] = (10, 10)
) -> None:
    """
    Plot all fractures as line segments in the xy-plane (top-down view).

    Parameters
    ----------
    centers : List[float]
        Flat list: [x0,y0,z0, x1,y1,z1, ...]
    thetas : List[float]
        Orientation angles in degrees (clockwise from +y).
    L : float
        Full length of each fracture.
    H : float
        Full height (ignored in top-down view).
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    """
    if len(centers) % 3 != 0:
        raise ValueError("centers must have length multiple of 3")

    Nf = len(centers) // 3
    if len(thetas) != Nf:
        raise ValueError("thetas length must match number of fractures")

    plt.figure(figsize=figsize)
    half_L = L / 2.0

    for i in range(Nf):
        cx = centers[3 * i]
        cy = centers[3 * i + 1]
        theta_rad = np.radians(thetas[i])  # clockwise from +y

        # Direction vector along strike (from center)
        dx = half_L * np.sin(theta_rad)   # +x when theta=90°
        dy = half_L * np.cos(theta_rad)   # +y when theta=0°

        x0 = cx - dx
        y0 = cy - dy
        x1 = cx + dx
        y1 = cy + dy

        plt.plot([x0, x1], [y0, y1], 'b-', linewidth=1.2)

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')  # critical: preserves aspect ratio
    plt.show()