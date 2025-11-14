import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import os

from ResFracPython.geometry import Point3D, Point2D
from ResFracPython.fracture import FractureElement, create_fracture_elements
from scipy.sparse import dok_matrix, csr_matrix
from ResFracPython.plotting import (
    plot_grid_with_fracture_counts, plot_fractures_topdown,
    plot_connection_matrix, generate_connection_report,
    plot_fractures_3d, _sparse_to_adjacency_list
)
from ResFracPython.testing.test_data import crossing_fractures, vertical_separation_fractures, intersecting_vertical_fractures
from ResFracPython.testing.random_fractures import generate_fracture_data
from ResFracPython.intersection import line_segment_intersects_rect, vectorized_line_rect_intersect
from ResFracPython.grid import initialize_grid
from ResFracPython.connections import find_fracture_connections

# Create output directory
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)


# Use randomly generated fracture data
num_fractures = 100
fracture_length = 200.0
fracture_height = 80.0
centers, thetas = generate_fracture_data(num_fractures, seed=42)
length = fracture_length
height = fracture_height


# Initialize grid with bounds computation, sizing, and fracture-cell intersection
grid_sparse_csr, nf, fractures, x_min, x_max, y_min, y_max, grid_size = initialize_grid(
    centers, thetas, length, height, padding_factor=1.0, cell_size_factor=0.25)

# Calculate cell size for plotting
cell_size = (x_max - x_min) / grid_size

# Plot top-down view without labels
plot_fractures_topdown(
    centers, thetas, length, height,
    title="Fractures (Top View)",
    x_min=x_min, 
    x_max=x_max,
    y_min=y_min, 
    y_max=y_max,
    show_labels=False,
    filename=os.path.join(output_dir, "fractures_topdown.png")
)

# Plot top-down view with labels
plot_fractures_topdown(
    centers, thetas, length, height,
    title="Fractures (Top View) - Labeled",
    x_min=x_min, 
    x_max=x_max,
    y_min=y_min, 
    y_max=y_max,
    show_labels=True,
    filename=os.path.join(output_dir, "fractures_topdown_labeled.png")
)

# Plot the grid with fracture counts (no labels)
plot_grid_with_fracture_counts(
    grid_sparse_csr=grid_sparse_csr,
    x_min=x_min,
    x_max=x_max,
    y_min=y_min,
    y_max=y_max,
    grid_size=grid_size,
    cell_size=cell_size,
    centers=centers,
    thetas=thetas,
    length=length,
    height=height,
    show_labels=False,
    filename=os.path.join(output_dir, "grid_fracture_counts.png")
)

# Plot the grid with fracture counts and labels
plot_grid_with_fracture_counts(
    grid_sparse_csr=grid_sparse_csr,
    x_min=x_min,
    x_max=x_max,
    y_min=y_min,
    y_max=y_max,
    grid_size=grid_size,
    cell_size=cell_size,
    centers=centers,
    thetas=thetas,
    length=length,
    height=height,
    show_labels=True,
    filename=os.path.join(output_dir, "grid_fracture_counts_labeled.png")
)

# Find all fracture connections using the connections module
connections, intersection_points = find_fracture_connections(
    grid_sparse_csr=grid_sparse_csr,
    fractures=fractures,
    grid_size=grid_size
)

# Generate and print connection report
print("\n")
report = generate_connection_report(
    connections=connections,
    intersection_points=intersection_points,
    filename=os.path.join(output_dir, "fracture_connections_report.txt")
)
print(report)

# Plot connection matrix
plot_connection_matrix(
    connections=connections,
    title=f"Fracture Connection Matrix ({nf} fractures)",
    filename=os.path.join(output_dir, "connection_matrix.png")
)

# Plot fractures in 3D perspective (without connections highlighted)
plot_fractures_3d(
    centers, thetas, length, height,
    title="Fractures (3D View)",
    filename=os.path.join(output_dir, "fractures_3d.png"),
    show_labels=False
)

# Plot fractures in 3D perspective with connections highlighted
plot_fractures_3d(
    centers, thetas, length, height,
    title="Fractures (3D View) - With Connections",
    filename=os.path.join(output_dir, "fractures_3d_connections.png"),
    connections=connections,
    show_labels=True
)
