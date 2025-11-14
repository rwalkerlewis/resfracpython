"""
Example usage of the runExample function.

This script demonstrates how to use the refactored runExample() function
to analyze fracture datasets with a clean, structured approach.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from demo import runExample
from ResFracPython.testing.test_data import (
    crossing_fractures,
    vertical_separation_fractures,
    intersecting_vertical_fractures
)


def main():
    """Run examples using different datasets."""
    
    print("\n" + "="*70)
    print("ResFracPython - Using runExample Function")
    print("="*70)
    
    # Example 1: Crossing fractures
    print("\n" + "-"*70)
    print("Example 1: Crossing Fractures (Baseline)")
    print("-"*70)
    
    centers, thetas, length, height = crossing_fractures()
    dataset_info = {
        'name': 'crossing_fractures',
        'centers': centers,
        'thetas': thetas,
        'length': length,
        'height': height
    }
    result = runExample(dataset_info)
    
    print("\nResult Summary:")
    print(f"  Output directory: {result['output_dir']}")
    print(f"  Number of fractures: {result['nf']}")
    print(f"  Total connections: {result['stats']['total_connections']}")
    
    # Example 2: Vertical separation (Z-axis screening demonstration)
    print("\n" + "-"*70)
    print("Example 2: Vertical Separation (Z-Axis Screening)")
    print("-"*70)
    
    centers, thetas, length, height = vertical_separation_fractures()
    dataset_info = {
        'name': 'vertical_separation',
        'centers': centers,
        'thetas': thetas,
        'length': length,
        'height': height
    }
    result = runExample(dataset_info)
    
    print("\nResult Summary:")
    print(f"  Output directory: {result['output_dir']}")
    print(f"  Number of fractures: {result['nf']}")
    print(f"  Total connections: {result['stats']['total_connections']}")
    print("  (Note: 0 connections due to vertical separation in z-axis)")
    
    # Example 3: Intersecting vertical fractures in 3D
    print("\n" + "-"*70)
    print("Example 3: 3D Intersecting Fractures")
    print("-"*70)
    
    centers, thetas, length, height = intersecting_vertical_fractures()
    dataset_info = {
        'name': 'intersecting_3d',
        'centers': centers,
        'thetas': thetas,
        'length': length,
        'height': height
    }
    result = runExample(dataset_info)
    
    print("\nResult Summary:")
    print(f"  Output directory: {result['output_dir']}")
    print(f"  Number of fractures: {result['nf']}")
    print(f"  Total connections: {result['stats']['total_connections']}")
    
    print("\n" + "="*70)
    print("All examples completed! Check output/ directory for results.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
