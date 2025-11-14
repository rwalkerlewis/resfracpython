"""
Test script to demonstrate vertical separation datasets with z-axis screening.
"""

import numpy as np
from testing.test_data import (
    crossing_fractures,
    vertical_separation_fractures,
    intersecting_vertical_fractures
)
from grid import initialize_grid
from fracture import FractureElement

def z_bounding_boxes_overlap(frac1: FractureElement, frac2: FractureElement) -> bool:
    """Check if two fractures have overlapping z-axis bounding boxes."""
    z_min1, z_max1 = frac1.get_z_range()
    z_min2, z_max2 = frac2.get_z_range()
    return z_max1 >= z_min2 and z_max2 >= z_min1

def test_dataset(name, centers, thetas, length, height):
    """Test a fracture dataset and report results."""
    print(f"\n{'='*70}")
    print(f"DATASET: {name}")
    print(f"{'='*70}")
    
    nf = len(centers) // 3
    print(f"Number of fractures: {nf}")
    print(f"Fracture length: {length}, height: {height}")
    print(f"\nFracture centers (x, y, z):")
    for i in range(nf):
        x, y, z = centers[3*i:3*i+3]
        z_min, z_max = centers[3*i+2] - height/2, centers[3*i+2] + height/2
        print(f"  F{i}: ({x:7.1f}, {y:7.1f}, {z:7.1f}) - z range: [{z_min:7.1f}, {z_max:7.1f}]")
    
    # Initialize grid
    grid_sparse_csr, nf_actual, fractures, x_min, x_max, y_min, y_max, grid_size = initialize_grid(
        centers, thetas, length, height, padding_factor=1.0
    )
    
    # Find connections with z-axis screening
    connections = [[] for _ in range(nf)]
    checked_pairs = set()
    
    for i in range(nf):
        for j in range(i+1, nf):
            pair = (i, j)
            if pair not in checked_pairs:
                checked_pairs.add(pair)
                z_overlap = z_bounding_boxes_overlap(fractures[i], fractures[j])
                xy_intersect = fractures[i].intersects(fractures[j])
                
                if z_overlap and xy_intersect:
                    connections[i].append(j)
                    connections[j].append(i)
    
    # Report results
    total_connections = sum(len(conns) for conns in connections) // 2
    connected = [i for i in range(nf) if connections[i]]
    
    print(f"\nGrid dimensions: {grid_size}x{grid_size} cells")
    print(f"Total connections found: {total_connections}")
    print(f"Connected fractures: {len(connected)}/{nf}")
    
    if total_connections > 0:
        print(f"\nConnections (with z-axis screening):")
        for i in range(nf):
            if connections[i]:
                print(f"  F{i}: {connections[i]}")
    else:
        print(f"\nNo connections found (all fractures are separated)")


if __name__ == "__main__":
    # Test 1: Crossing fractures (baseline)
    centers1, thetas1, length1, height1 = crossing_fractures()
    test_dataset("Crossing Fractures (Baseline)", centers1, thetas1, length1, height1)
    
    # Test 2: Vertical separation
    centers2, thetas2, length2, height2 = vertical_separation_fractures()
    test_dataset("Vertical Separation (No Intersections)", centers2, thetas2, length2, height2)
    
    # Test 3: Intersecting vertical fractures
    centers3, thetas3, length3, height3 = intersecting_vertical_fractures()
    test_dataset("Intersecting Vertical Fractures", centers3, thetas3, length3, height3)
    
    print(f"\n{'='*70}")
