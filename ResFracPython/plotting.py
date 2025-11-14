"""
Plotting utilities for fracture visualization and analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from typing import Tuple, List, Union


def _sparse_to_adjacency_list(connections_matrix: csr_matrix) -> List[List[int]]:
    """
    Convert sparse CSR connection matrix to adjacency list format.
    
    Parameters
    ----------
    connections_matrix : csr_matrix
        Sparse matrix where entry (i,j)=1 if fractures i and j are connected.
    
    Returns
    -------
    List[List[int]]
        Adjacency list where connections_list[i] contains indices of fractures
        connected to fracture i.
    """
    nf = connections_matrix.shape[0]
    connections_list = [[] for _ in range(nf)]
    
    for i in range(nf):
        # Get non-zero column indices for row i
        row_data = connections_matrix.getrow(i)
        connected_to_i = row_data.nonzero()[1]
        connections_list[i] = sorted(connected_to_i.tolist())
    
    return connections_list


def plot_connection_matrix(
    connections: Union[List[List[int]], csr_matrix],
    figsize: Tuple[float, float] = None,
    title: str = "Fracture Connection Matrix",
    filename: str = "connection_matrix.png"
) -> None:
    """
    Plot a connection matrix where blocks are colored by the y-axis (row) fracture ID.
    
    Parameters
    ----------
    connections : List[List[int]] or csr_matrix
        Either a list where connections[i] contains the indices of fractures that 
        intersect fracture i, or a sparse CSR matrix where entry (i,j)=1 if 
        fractures i and j are connected.
    figsize : tuple, optional
        Figure size (width, height). Auto-sized based on fracture count if None.
    title : str, optional
        Plot title. Default is "Fracture Connection Matrix".
    filename : str, optional
        Output filename. Default is "connection_matrix.png".
    """
    # Convert sparse matrix to adjacency list if needed
    if isinstance(connections, csr_matrix):
        connections = _sparse_to_adjacency_list(connections)
    
    nf = len(connections)
    
    # Create a matrix where each cell is colored by its row (y-axis) fracture ID
    # Values: 0 for no connection, 1-nf for connection colored by y-axis ID
    color_matrix = np.zeros((nf, nf), dtype=int)
    for i in range(nf):
        for j in connections[i]:
            # Color the cell with the y-axis fracture ID (row index i)
            color_matrix[i, j] = i + 1  # +1 to distinguish from 0 (no connection)
    
    # Auto-size figure based on number of fractures
    if figsize is None:
        # Aim for ~0.3 inches per fracture, minimum 8, maximum 16
        size = max(8, min(16, nf * 0.3))
        figsize = (size, size)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display heatmap with a perceptually uniform colormap
    # 0 = white (no connection), 1-nf = colors scaled from colormap
    # Create a colormap that starts with white for 0, then uses tab20 or hsv for values 1-nf
    from matplotlib.colors import ListedColormap
    if nf <= 20:
        base_cmap = plt.cm.tab20
    else:
        base_cmap = plt.cm.hsv
    colors = ['white'] + [base_cmap(i / max(nf - 1, 1)) for i in range(nf)]
    custom_cmap = ListedColormap(colors)
    im = ax.imshow(color_matrix, cmap=custom_cmap, aspect='auto', origin='upper', vmin=0, vmax=nf)
    
    # Set ticks and labels - sparse for large matrices
    if nf <= 50:
        ax.set_xticks(np.arange(nf))
        ax.set_yticks(np.arange(nf))
        ax.set_xticklabels(np.arange(nf), fontsize=8)
        ax.set_yticklabels(np.arange(nf), fontsize=8)
    else:
        # For large matrices, show ticks at regular intervals
        tick_interval = max(1, nf // 10)  # Show ~10 ticks
        ticks = np.arange(0, nf, tick_interval)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(ticks, fontsize=8)
        ax.set_yticklabels(ticks, fontsize=8)
    
    # Rotate the tick labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add colorbar with y-axis fracture IDs
    cbar = plt.colorbar(im, ax=ax, label='Y-Axis Fracture ID', pad=0.02)
    cbar.set_label('Y-Axis Fracture ID (color = connected to which fracture row)', fontsize=9)
    
    # Add grid for clarity - only if small enough
    if nf <= 100:
        ax.set_xticks(np.arange(nf) - 0.5, minor=True)
        ax.set_yticks(np.arange(nf) - 0.5, minor=True)
        ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3, alpha=0.5)
    
    # Set labels
    ax.set_xlabel('X-Axis Fracture ID (connects to)', fontsize=10)
    ax.set_ylabel('Y-Axis Fracture ID (color scale)', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Add annotations only if small enough to be readable
    if nf <= 15:
        for i in range(nf):
            for j in range(nf):
                if color_matrix[i, j] > 0:
                    text = ax.text(j, i, '●', ha="center", va="center",
                                 color="white", fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved connection matrix plot to {filename}")
    plt.close()


def generate_connection_report(
    connections: Union[List[List[int]], csr_matrix],
    intersection_points: dict = None,
    filename: str = "fracture_connections_report.txt"
) -> str:
    """
    Generate a detailed text report of fracture connections.
    
    Parameters
    ----------
    connections : List[List[int]] or csr_matrix
        Either a list where connections[i] contains the indices of fractures that 
        intersect fracture i, or a sparse CSR matrix where entry (i,j)=1 if 
        fractures i and j are connected.
    intersection_points : dict, optional
        Dictionary with keys (i, j) tuples and Point3D values for intersection coordinates.
    filename : str, optional
        Output filename for the report. Default is "fracture_connections_report.txt".
    
    Returns
    -------
    str
        The report text that was written to file and can be printed.
    """
    # Convert sparse matrix to adjacency list if needed
    if isinstance(connections, csr_matrix):
        connections = _sparse_to_adjacency_list(connections)
    
    if intersection_points is None:
        intersection_points = {}
    
    nf = len(connections)
    
    # Calculate statistics
    total_connections = sum(len(conns) for conns in connections) // 2
    connected_fractures = sum(1 for conns in connections if len(conns) > 0)
    unconnected_fractures = nf - connected_fractures
    
    # Find fractures with most connections
    connections_per_fracture = [(i, len(conns)) for i, conns in enumerate(connections)]
    connections_per_fracture.sort(key=lambda x: x[1], reverse=True)
    
    # Build report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("FRACTURE CONNECTION REPORT")
    report_lines.append("=" * 70)
    report_lines.append("")
    
    # Summary statistics
    report_lines.append("SUMMARY STATISTICS")
    report_lines.append("-" * 70)
    report_lines.append(f"Total number of fractures: {nf}")
    report_lines.append(f"Total unique connections: {total_connections}")
    report_lines.append(f"Connected fractures: {connected_fractures} ({100*connected_fractures/nf:.1f}%)")
    report_lines.append(f"Unconnected fractures: {unconnected_fractures} ({100*unconnected_fractures/nf:.1f}%)")
    if total_connections > 0:
        avg_connections = 2 * total_connections / connected_fractures if connected_fractures > 0 else 0
        report_lines.append(f"Average connections per connected fracture: {avg_connections:.2f}")
    report_lines.append("")
    
    # Fractures with most connections
    report_lines.append("TOP FRACTURES BY CONNECTION COUNT")
    report_lines.append("-" * 70)
    if connections_per_fracture:
        max_count = max(count for _, count in connections_per_fracture)
        for frac_id, count in connections_per_fracture[:min(10, nf)]:
            if count > 0:
                report_lines.append(f"  Fracture {frac_id:4d}: {count:3d} connection{'s' if count != 1 else ' '}")
    report_lines.append("")
    
    # Detailed connections
    report_lines.append("DETAILED CONNECTIONS BY FRACTURE")
    report_lines.append("-" * 70)
    for i in range(nf):
        if connections[i]:
            for j in connections[i]:
                if i < j:  # Only show each connection once (in canonical order)
                    pair = (i, j)
                    if pair in intersection_points:
                        pt = intersection_points[pair]
                        report_lines.append(f"  Fracture {i:4d} ↔ Fracture {j:4d}: ({pt.x:10.2f}, {pt.y:10.2f}, {pt.z:10.2f})")
                    else:
                        report_lines.append(f"  Fracture {i:4d} ↔ Fracture {j:4d}")
    
    if unconnected_fractures > 0:
        report_lines.append("")
        report_lines.append("UNCONNECTED FRACTURES")
        report_lines.append("-" * 70)
        unconnected_ids = [i for i in range(nf) if not connections[i]]
        unconnected_str = ", ".join(str(i) for i in unconnected_ids)
        if len(unconnected_ids) <= 20:
            report_lines.append(f"  {unconnected_str}")
        else:
            report_lines.append(f"  {unconnected_ids[0]}, {unconnected_ids[1]}, ... , {unconnected_ids[-2]}, {unconnected_ids[-1]}")
            report_lines.append(f"  ({len(unconnected_ids)} total unconnected fractures)")
    
    report_lines.append("")
    report_lines.append("=" * 70)
    
    # Join and return
    report_text = "\n".join(report_lines)
    
    # Write to file
    with open(filename, 'w') as f:
        f.write(report_text)
    
    return report_text


def plot_fractures_topdown(
    centers: np.ndarray,
    thetas: np.ndarray,
    length: float,
    height: float,
    title: str = "Fractures (Top-Down View)",
    figsize: Tuple[float, float] = (10, 10),
    x_min: float = None,
    x_max: float = None,
    y_min: float = None,
    y_max: float = None,
    show_labels: bool = False,
    filename: str = None
) -> None:
    """
    Plot all fractures as line segments in the xy-plane (top-down view).

    Parameters
    ----------
    centers : np.ndarray
        Flat array: [x0,y0,z0, x1,y1,z1, ...]
    thetas : np.ndarray
        Orientation angles in degrees (clockwise from +y).
    length : float
        Full length of each fracture.
    height : float
        Full height (ignored in top-down view).
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    x_min, x_max, y_min, y_max : float, optional
        Explicit axis limits. If None, auto-scales to data.
    show_labels : bool, optional
        If True, display fracture ID labels with arrows. Default is False.
    filename : str, optional
        Output filename. If None, auto-generates based on show_labels.
    """
    if len(centers) % 3 != 0:
        raise ValueError("centers must have length multiple of 3")

    Nf = len(centers) // 3
    if len(thetas) != Nf:
        raise ValueError("thetas length must match number of fractures")

    plt.figure(figsize=figsize)
    half_L = length / 2.0

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
        
        if show_labels:
            # Add offset label with arrow pointing to fracture plane
            # Offset the label by a small distance (perpendicular to fracture strike)
            offset_dist = half_L * 0.4  # offset distance from center
            offset_x = offset_dist * np.cos(theta_rad)   # perpendicular to strike
            offset_y = -offset_dist * np.sin(theta_rad)  # perpendicular to strike
            
            label_x = cx + offset_x
            label_y = cy + offset_y
            
            # Arrow endpoint: find closest point on fracture line to label
            # Project label onto fracture line
            t = ((label_x - cx) * dx + (label_y - cy) * dy) / (dx**2 + dy**2 + 1e-10)
            t = np.clip(t, -1, 1)  # clamp to fracture endpoints
            arrow_x = cx + t * dx
            arrow_y = cy + t * dy
            
            # Draw arrow from label to fracture plane
            plt.annotate('', xy=(arrow_x, arrow_y), xytext=(label_x, label_y),
                        arrowprops=dict(arrowstyle='->', lw=1.0, color='red', alpha=0.7))
            
            # Add fracture ID label
            plt.text(label_x, label_y, str(i), ha='center', va='center', 
                    fontsize=8, fontweight='bold', color='red',
                    bbox=dict(boxstyle='circle,pad=0.3', facecolor='yellow', alpha=0.8))

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')  # critical: preserves aspect ratio

    # Set axis limits AFTER axis('equal') so they are not overridden
    if x_min is not None and x_max is not None:
        plt.xlim(x_min, x_max)
    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)

    plt.tight_layout()

    # Save to file instead of showing
    if filename is None:
        filename = 'fractures_topdown.png'
        if show_labels:
            filename = 'fractures_topdown_labeled.png'
    plt.savefig(filename, dpi=150)
    print(f"Saved fracture plot to {filename}")
    plt.close()


def plot_grid_with_fracture_counts(
    grid_sparse_csr: csr_matrix,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    grid_size: int,
    cell_size: float = None,
    figsize: tuple = (12, 10),
    title: str = "Spatial Grid Fracture Count",
    centers: np.ndarray = None,
    thetas: np.ndarray = None,
    length: float = None,
    height: float = None,
    show_labels: bool = False,
    skip_fracture_overlay: bool = False,
    filename: str = None
) -> None:
    """
    Plot a spatial grid with numbers indicating how many fractures fall within each cell.
    Optionally overlay actual fractures as translucent lines.
    
    Parameters
    ----------
    grid_sparse_csr : csr_matrix
        Sparse matrix of shape (num_cells, nf) where entry (i, j) indicates
        whether fracture j intersects grid cell i.
    x_min, x_max : float
        X-axis bounds of the grid.
    y_min, y_max : float
        Y-axis bounds of the grid.
    grid_size : int
        Number of cells per side (grid_size x grid_size).
    cell_size : float, optional
        Cell size for axis labeling.
    figsize : tuple, optional
        Figure size (width, height). Default is (12, 10).
    title : str, optional
        Plot title. Default is "Spatial Grid Fracture Count".
    centers : np.ndarray, optional
        Flat array [x0,y0,z0, x1,y1,z1, ...] for fracture centers. If provided,
        fractures will be overlaid as translucent lines (unless skip_fracture_overlay=True).
    thetas : np.ndarray, optional
        Orientation angles in degrees (clockwise from +y). Required if centers is provided.
    length : float, optional
        Full length of each fracture. Required if centers is provided.
    height : float, optional
        Full height (ignored in top-down view). Not used if centers is provided.
    show_labels : bool, optional
        If True, display fracture ID labels with arrows. Default is False.
    skip_fracture_overlay : bool, optional
        If True, skip overlaying fracture lines (useful for very large datasets). Default is False.
    filename : str, optional
        Output filename. If None, auto-generates based on show_labels.
    """
    num_cells = grid_size * grid_size
    nf = grid_sparse_csr.shape[1]
    
    # Calculate cell dimensions from grid bounds
    dx = (x_max - x_min) / grid_size
    dy = (y_max - y_min) / grid_size
    
    # Convert sparse matrix to dense to get fracture counts per cell
    grid_dense = grid_sparse_csr.toarray()  # Shape: (num_cells, nf)
    fractures_per_cell = np.sum(grid_dense, axis=1).astype(int)  # Sum across fractures for each cell
    
    # Reshape fractures_per_cell into a 2D grid for efficient visualization
    grid_2d = fractures_per_cell.reshape((grid_size, grid_size))
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare colormap and normalizer (adaptive to actual max count)
    vmax = max(1, int(np.max(fractures_per_cell)))
    # Use a colormap where white is 0, and green is high counts
    cmap = plt.cm.Greens
    norm = plt.Normalize(vmin=0, vmax=vmax)
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    
    # Use imshow for fast rendering instead of drawing individual rectangles
    # Note: imshow shows y-axis inverted by default, so we adjust origin
    im = ax.imshow(grid_2d.T, extent=[x_min, x_max, y_min, y_max], 
                   origin='lower', cmap=cmap, norm=norm, aspect='auto')
    
    # Add grid lines for cell boundaries (sparse grid lines for large grids)
    if grid_size <= 50:
        # For smaller grids, show all grid lines
        for i in range(grid_size + 1):
            ax.axhline(y=y_min + i * dy, color='black', linewidth=0.5, alpha=0.3)
            ax.axvline(x=x_min + i * dx, color='black', linewidth=0.5, alpha=0.3)
    else:
        # For larger grids, show sparse grid lines (every nth line)
        step = max(1, grid_size // 20)  # Show ~20 lines in each direction
        for i in range(0, grid_size + 1, step):
            ax.axhline(y=y_min + i * dy, color='black', linewidth=0.5, alpha=0.3)
            ax.axvline(x=x_min + i * dx, color='black', linewidth=0.5, alpha=0.3)
    
    # Add text labels for cells with multiple fractures (likely intersection points)
    # Use higher threshold since we only label cells with count > 1 (sparse)
    if grid_size <= 100:  # Can handle larger grids since we only label multi-fracture cells
        for i in range(grid_size):
            for j in range(grid_size):
                count = grid_2d[i, j]
                if count > 1:  # Label cells with multiple fractures (potential intersections)
                    x_text = x_min + (i + 0.5) * dx
                    y_text = y_min + (j + 0.5) * dy
                    # Black text without background to preserve cell color visibility
                    ax.text(x_text, y_text, str(count), ha='center', va='center',
                           fontsize=10, fontweight='bold', color='black',
                           clip_on=True, zorder=5)
    
    # Overlay fractures as translucent lines if data is provided
    if (centers is not None and thetas is not None and length is not None 
        and not skip_fracture_overlay):
        if len(centers) % 3 != 0:
            raise ValueError("centers must have length multiple of 3")
        
        Nf = len(centers) // 3
        if len(thetas) != Nf:
            raise ValueError("thetas length must match number of fractures")
        
        half_L = length / 2.0
        
        for i in range(Nf):
            cx = centers[3 * i]
            cy = centers[3 * i + 1]
            theta_rad = np.radians(thetas[i])  # clockwise from +y
            
            # Direction vector along strike (from center)
            dx_frac = half_L * np.sin(theta_rad)   # +x when theta=90°
            dy_frac = half_L * np.cos(theta_rad)   # +y when theta=0°
            
            x0 = cx - dx_frac
            y0 = cy - dy_frac
            x1 = cx + dx_frac
            y1 = cy + dy_frac
            
            # Plot with translucency (alpha=0.3) and distinct color
            ax.plot([x0, x1], [y0, y1], 'b-', linewidth=2.0, alpha=0.3, label='Fractures' if i == 0 else '')
            
            if show_labels:
                # Add offset label with arrow pointing to fracture plane
                # Offset the label by a small distance (perpendicular to fracture strike)
                offset_dist = half_L * 0.35
                offset_x = offset_dist * np.cos(theta_rad)   # perpendicular to strike
                offset_y = -offset_dist * np.sin(theta_rad)  # perpendicular to strike
                
                label_x = cx + offset_x
                label_y = cy + offset_y
                
                # Arrow endpoint: find closest point on fracture line to label
                # Project label onto fracture line
                t = ((label_x - cx) * dx_frac + (label_y - cy) * dy_frac) / (dx_frac**2 + dy_frac**2 + 1e-10)
                t = np.clip(t, -1, 1)  # clamp to fracture endpoints
                arrow_x = cx + t * dx_frac
                arrow_y = cy + t * dy_frac
                
                # Draw arrow from label to fracture plane
                ax.annotate('', xy=(arrow_x, arrow_y), xytext=(label_x, label_y),
                           arrowprops=dict(arrowstyle='->', lw=0.8, color='red', alpha=0.7))
                
                # Add fracture ID label
                ax.text(label_x, label_y, str(i), ha='center', va='center',
                       fontsize=7, fontweight='bold', color='red',
                       bbox=dict(boxstyle='circle,pad=0.2', facecolor='yellow', alpha=0.8),
                       clip_on=True, zorder=10)
    
    # Set aspect ratio after all patches and lines are added
    ax.set_aspect('equal')
    
    # Set axis labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'{title}\n(Grid size: {grid_size}x{grid_size}, Numbers = fractures per cell)')
    
    # Add legend if fractures were overlaid
    if centers is not None and thetas is not None and length is not None:
        ax.legend(loc='upper right')
    
    # Add colorbar with integer ticks (whole numbers)
    cbar = plt.colorbar(mappable, ax=ax, label='Fracture Count')
    # Set integer tick locator
    try:
        from matplotlib.ticker import MaxNLocator
        cbar.locator = MaxNLocator(integer=True)
        cbar.update_ticks()
    except Exception:
        # Fallback: set ticks explicitly
        cbar.set_ticks(np.arange(0, vmax + 1))
    
    plt.tight_layout()
    
    # Save to file instead of showing
    if filename is None:
        filename = 'grid_fracture_counts.png'
        if show_labels:
            filename = 'grid_fracture_counts_labeled.png'
    plt.savefig(filename, dpi=150)
    print(f"Saved grid plot to {filename}")
    plt.close()
    
    # Print statistics
    print(f"\nGrid Statistics:")
    print(f"  Grid dimensions: {grid_size} x {grid_size} = {num_cells} cells")
    print(f"  Number of fractures: {nf}")
    print(f"  Sparse matrix shape: {grid_sparse_csr.shape}")
    print(f"  Non-zero entries: {grid_sparse_csr.nnz}")
    print(f"  Sparsity: {1 - grid_sparse_csr.nnz / (num_cells * nf):.2%}")
    print(f"  Cells with fractures: {np.count_nonzero(fractures_per_cell)}")
    print(f"  Max fractures in single cell: {np.max(fractures_per_cell)}")
    print(f"  Min fractures in single cell: {np.min(fractures_per_cell)}")


def plot_fractures_3d(
    centers: np.ndarray,
    thetas: np.ndarray,
    length: float,
    height: float,
    title: str = "Fractures (3D View)",
    figsize: Tuple[float, float] = (14, 10),
    filename: str = None,
    connections: Union[List[List[int]], csr_matrix] = None,
    show_labels: bool = False
) -> None:
    """
    Plot all fractures in 3D perspective showing x, y, and z coordinates.
    
    Parameters
    ----------
    centers : np.ndarray
        Flat array: [x0,y0,z0, x1,y1,z1, ...]
    thetas : np.ndarray
        Orientation angles in degrees (clockwise from +y).
    length : float
        Full length of each fracture.
    height : float
        Full height of each fracture (extends in z direction).
    title : str, optional
        Plot title. Default is "Fractures (3D View)".
    figsize : tuple, optional
        Figure size (width, height). Default is (14, 10).
    filename : str, optional
        Output filename. If None, plot is shown instead of saved.
    connections : List[List[int]] or csr_matrix, optional
        Fracture connections for highlighting intersecting pairs. If provided,
        connected fractures will be drawn with thicker lines.
    show_labels : bool, optional
        If True, display fracture ID labels near their centers. Default is False.
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Convert sparse matrix to adjacency list if needed
    if isinstance(connections, csr_matrix):
        connections = _sparse_to_adjacency_list(connections)
    
    if len(centers) % 3 != 0:
        raise ValueError("centers must have length multiple of 3")
    
    Nf = len(centers) // 3
    if len(thetas) != Nf:
        raise ValueError("thetas length must match number of fractures")
    
    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    half_L = length / 2.0
    half_H = height / 2.0
    
    # Define colors for fractures
    colors = plt.cm.tab20(np.linspace(0, 1, max(20, Nf)))
    
    # Track which fractures are connected for visualization
    connected_fractures = set()
    if connections is not None:
        for i in range(Nf):
            if connections[i]:
                connected_fractures.add(i)
                for j in connections[i]:
                    connected_fractures.add(j)
    
    # Plot each fracture as a rectangular plane in 3D
    for i in range(Nf):
        cx = centers[3 * i]
        cy = centers[3 * i + 1]
        cz = centers[3 * i + 2]
        theta_rad = np.radians(thetas[i])
        
        # Direction vector along strike (in xy plane)
        dx = half_L * np.sin(theta_rad)
        dy = half_L * np.cos(theta_rad)
        
        # Fracture corners: 4 corners of the rectangular plane
        # Along strike direction (xy) and vertical direction (z)
        corners = np.array([
            [cx - dx, cy - dy, cz - half_H],  # bottom-back
            [cx + dx, cy + dy, cz - half_H],  # bottom-front
            [cx + dx, cy + dy, cz + half_H],  # top-front
            [cx - dx, cy - dy, cz + half_H],  # top-back
        ])
        
        # Close the rectangle
        corners = np.vstack([corners, corners[0]])
        
        # Determine line properties based on connectivity
        is_connected = i in connected_fractures if connections else False
        linewidth = 2.5 if is_connected else 1.5
        alpha = 0.8 if is_connected else 0.6
        color = colors[i % len(colors)]
        
        # Plot fracture as a wireframe rectangle
        ax.plot(corners[:, 0], corners[:, 1], corners[:, 2],
               color=color, linewidth=linewidth, alpha=alpha, label=f'F{i}')
        
        # Also draw two diagonal lines to show the plane orientation
        ax.plot([corners[0, 0], corners[2, 0]], 
               [corners[0, 1], corners[2, 1]], 
               [corners[0, 2], corners[2, 2]],
               color=color, linewidth=0.5, alpha=0.3, linestyle='--')
        
        # Add label if requested
        if show_labels:
            ax.text(cx, cy, cz, f'F{i}', fontsize=8, fontweight='bold')
    
    # Set labels and title
    ax.set_xlabel('X', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y', fontsize=11, fontweight='bold')
    ax.set_zlabel('Z', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Set viewing angle for better perspective
    ax.view_init(elev=20, azim=45)
    
    # Add legend if there aren't too many fractures
    if Nf <= 20:
        ax.legend(loc='upper left', fontsize=8, ncol=2)
    
    plt.tight_layout()
    
    # Save or show
    if filename is None:
        filename = 'fractures_3d.png'
    
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved 3D fracture plot to {filename}")
    plt.close()
