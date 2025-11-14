"""
ResFracPython Demo Script

This script demonstrates the basic functionality of the ResFracPython framework.
It loads different test datasets, analyzes fracture intersections, and generates
visualizations and reports.
"""

import os
from ResFracPython.testing.test_data import (
    crossing_fractures,
    vertical_separation_fractures,
    intersecting_vertical_fractures
)
from ResFracPython.grid import initialize_grid
from ResFracPython.connections import find_fracture_connections
from ResFracPython.plotting import (
    plot_fractures_topdown,
    plot_grid_with_fracture_counts,
    plot_connection_matrix,
    plot_fractures_3d,
    generate_connection_report,
    _sparse_to_adjacency_list
)


def runExample(dataset_info):
    """
    Run complete fracture analysis example with plotting and reporting.
    
    This function encapsulates the entire workflow: grid initialization, connection
    detection, visualization generation, and report generation.
    
    Parameters
    ----------
    dataset_info : dict
        Dictionary containing:
        - 'name': str - Name of the dataset (used for output directory)
        - 'centers': np.ndarray - Flat array [x0,y0,z0, x1,y1,z1, ...]
        - 'thetas': np.ndarray - Orientation angles in degrees
        - 'length': float - Fracture length
        - 'height': float - Fracture height
    
    Returns
    -------
    dict
        Dictionary containing analysis results:
        - 'output_dir': str - Output directory path
        - 'nf': int - Number of fractures
        - 'connections': sparse matrix - Connection matrix
        - 'intersection_points': dict - Intersection coordinates
        - 'grid_info': dict - Grid dimensions and bounds
        - 'stats': dict - Summary statistics
    
    Example
    -------
    >>> from ResFracPython.testing.test_data import crossing_fractures
    >>> centers, thetas, length, height = crossing_fractures()
    >>> dataset_info = {
    ...     'name': 'my_dataset',
    ...     'centers': centers,
    ...     'thetas': thetas,
    ...     'length': length,
    ...     'height': height
    ... }
    >>> result = runExample(dataset_info)
    """
    # Extract dataset info
    dataset_name = dataset_info['name']
    centers = dataset_info['centers']
    thetas = dataset_info['thetas']
    length = dataset_info['length']
    height = dataset_info['height']
    
    # Create output directory
    output_dir = f"output/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Analyzing dataset: {dataset_name}")
    print(f"{'='*70}")
    
    # Initialize grid
    grid_sparse_csr, nf, fractures, x_min, x_max, y_min, y_max, grid_size = initialize_grid(
        centers, thetas, length, height, padding_factor=1.0
    )
    
    # Calculate cell size for plotting
    cell_size = (x_max - x_min) / grid_size
    
    grid_info = {
        'grid_size': grid_size,
        'num_cells': grid_size * grid_size,
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'cell_size': cell_size,
        'sparse_matrix': grid_sparse_csr
    }
    
    print(f"\nGrid created:")
    print(f"  Dimensions: {grid_info['grid_size']} x {grid_info['grid_size']} = {grid_info['num_cells']} cells")
    print(f"  Fractures: {nf}")
    print(f"  Domain: x=[{x_min:.1f}, {x_max:.1f}], y=[{y_min:.1f}, {y_max:.1f}]")
    
    # Generate visualizations
    print(f"\nGenerating visualizations...")
    _generate_visualizations(
        dataset_name, output_dir,
        centers, thetas, length, height,
        grid_info, nf
    )
    
    # Find connections
    print(f"Detecting fracture connections...")
    connections, intersection_points = find_fracture_connections(
        grid_sparse_csr=grid_sparse_csr,
        fractures=fractures,
        grid_size=grid_size
    )
    
    # Create connection-dependent visualizations
    _create_connection_visualizations(
        dataset_name, output_dir,
        centers, thetas, length, height,
        connections
    )
    print(f"  ✓ Generated 7 visualizations")
    print(f"Generating connection report...")
    report = generate_connection_report(
        connections=connections,
        intersection_points=intersection_points,
        filename=os.path.join(output_dir, "connection_report.txt")
    )
    print(f"  ✓ Saved to {os.path.join(output_dir, 'connection_report.txt')}")
    
    # Calculate statistics
    connections_list = _sparse_to_adjacency_list(connections)
    total_connections = sum(len(conns) for conns in connections_list) // 2
    connected_fracs = sum(1 for conns in connections_list if len(conns) > 0)
    
    stats = {
        'num_fractures': nf,
        'total_connections': total_connections,
        'connected_fractures': connected_fracs,
        'unconnected_fractures': nf - connected_fracs,
        'avg_connections': 2 * total_connections / connected_fracs if connected_fracs > 0 else 0
    }
    
    # Print summary
    print(f"\nConnection Summary:")
    print(f"  Total fractures: {stats['num_fractures']}")
    print(f"  Total connections: {stats['total_connections']}")
    print(f"  Connected fractures: {stats['connected_fractures']}")
    if stats['total_connections'] > 0:
        print(f"  Average connections per connected fracture: {stats['avg_connections']:.2f}")
    
    # Return comprehensive result
    return {
        'output_dir': output_dir,
        'nf': nf,
        'connections': connections,
        'intersection_points': intersection_points,
        'grid_info': grid_info,
        'stats': stats
    }


def _generate_visualizations(dataset_name, output_dir, centers, thetas, length, height, grid_info, nf):
    """
    Generate all visualizations for a fracture dataset.
    
    Parameters
    ----------
    dataset_name : str
        Name of dataset for titles
    output_dir : str
        Output directory path
    centers : np.ndarray
        Fracture centers
    thetas : np.ndarray
        Fracture orientations
    length : float
        Fracture length
    height : float
        Fracture height
    grid_info : dict
        Grid information (bounds, size, sparse matrix)
    nf : int
        Number of fractures
    """
    x_min = grid_info['x_min']
    x_max = grid_info['x_max']
    y_min = grid_info['y_min']
    y_max = grid_info['y_max']
    grid_size = grid_info['grid_size']
    cell_size = grid_info['cell_size']
    grid_sparse_csr = grid_info['sparse_matrix']
    
    # Top-down views
    plot_fractures_topdown(
        centers, thetas, length, height,
        title=f"{dataset_name} - Top View",
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        show_labels=False,
        filename=os.path.join(output_dir, "01_topdown.png")
    )
    
    plot_fractures_topdown(
        centers, thetas, length, height,
        title=f"{dataset_name} - Top View (Labeled)",
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        show_labels=True,
        filename=os.path.join(output_dir, "02_topdown_labeled.png")
    )
    
    # Grid analysis views
    plot_grid_with_fracture_counts(
        grid_sparse_csr=grid_sparse_csr,
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        grid_size=grid_size, cell_size=cell_size,
        centers=centers, thetas=thetas, length=length, height=height,
        show_labels=False,
        filename=os.path.join(output_dir, "03_grid.png")
    )
    
    plot_grid_with_fracture_counts(
        grid_sparse_csr=grid_sparse_csr,
        x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
        grid_size=grid_size, cell_size=cell_size,
        centers=centers, thetas=thetas, length=length, height=height,
        show_labels=True,
        filename=os.path.join(output_dir, "04_grid_labeled.png")
    )
    
    # 3D visualization (connections plotted separately after detection)
    plot_fractures_3d(
        centers, thetas, length, height,
        title=f"{dataset_name} - 3D View",
        filename=os.path.join(output_dir, "06_3d_view.png"),
        show_labels=False
    )
    
    # Placeholder for connection matrix and 3D with connections
    # (created after connections are detected in runExample)
    # Files 05 and 07 are generated in runExample after connection detection


def _create_connection_visualizations(dataset_name, output_dir, centers, thetas, length, height, connections):
    """
    Create visualizations that require connection information.
    
    Parameters
    ----------
    dataset_name : str
        Name of dataset for titles
    output_dir : str
        Output directory path
    centers : np.ndarray
        Fracture centers
    thetas : np.ndarray
        Fracture orientations
    length : float
        Fracture length
    height : float
        Fracture height
    connections : sparse matrix
        Connection matrix
    """
    # Connection matrix visualization
    plot_connection_matrix(
        connections=connections,
        title=f"{dataset_name} - Connection Matrix",
        filename=os.path.join(output_dir, "05_connection_matrix.png")
    )
    
    # 3D view with connections highlighted
    plot_fractures_3d(
        centers, thetas, length, height,
        title=f"{dataset_name} - 3D View (with Connections)",
        filename=os.path.join(output_dir, "07_3d_connections.png"),
        connections=connections,
        show_labels=True
    )


def analyze_fracture_dataset(centers, thetas, length, height, dataset_name):
    """
    Analyze a fracture dataset (legacy wrapper for runExample).
    
    Deprecated: Use runExample() instead with dataset_info dictionary.
    
    Parameters
    ----------
    centers : np.ndarray
        Flat array of fracture centers: [x0,y0,z0, x1,y1,z1, ...]
    thetas : np.ndarray
        Array of fracture orientations in degrees
    length : float
        Fracture length
    height : float
        Fracture height
    dataset_name : str
        Name of the dataset (used for output directory)
    
    Returns
    -------
    str
        Output directory path
    """
    dataset_info = {
        'name': dataset_name,
        'centers': centers,
        'thetas': thetas,
        'length': length,
        'height': height
    }
    result = runExample(dataset_info)
    return result['output_dir']


def main():
    """Run demo with all available test datasets."""
    
    print("\n" + "="*70)
    print("ResFracPython - Fracture Network Analysis Demo")
    print("="*70)
    
    # Test Dataset 1: Crossing fractures (baseline)
    centers, thetas, length, height = crossing_fractures()
    analyze_fracture_dataset(centers, thetas, length, height, "crossing_fractures")
    
    # Test Dataset 2: Vertical separation (z-axis screening demo)
    centers, thetas, length, height = vertical_separation_fractures()
    analyze_fracture_dataset(centers, thetas, length, height, "vertical_separation")
    
    # Test Dataset 3: 3D intersecting fractures
    centers, thetas, length, height = intersecting_vertical_fractures()
    analyze_fracture_dataset(centers, thetas, length, height, "intersecting_3d")
    
    print("\n" + "="*70)
    print("Demo completed! Check the output/ directory for results.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
