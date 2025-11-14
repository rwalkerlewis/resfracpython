# ResFracPython

A Python framework for 3D fracture network analysis, intersection detection, and visualization.

## Overview

ResFracPython analyzes vertical rectangular fractures in 3D space, detecting intersections between fractures and generating comprehensive reports and visualizations. The system uses efficient spatial partitioning (grid-based indexing) and hierarchical intersection detection to handle large fracture networks.

### Key Features

- **3D Fracture Representation**: Models vertical rectangular fractures with position, orientation, length, and height
- **Efficient Intersection Detection**: Uses z-axis screening and cross-product line segment testing
- **Spatial Grid Indexing**: Partitions the domain into a sparse grid for fast neighbor detection
- **3D Visualization**: Multiple visualization modes including top-down, 3D perspective, grid analysis, and connection matrices
- **Connection Reporting**: Generates detailed reports with intersection coordinates for each fracture pair
- **Modular Architecture**: Clean separation between core geometry, grid management, connection detection, and visualization

## Installation

### Requirements
- Python 3.9+
- NumPy
- Matplotlib
- SciPy

### Setup

```bash
# Clone the repository
git clone https://github.com/rwalkerlewis/resfracpython.git
cd resfracpython

# Install dependencies
pip install numpy matplotlib scipy
```

## Quick Start

```python
from main import *

# Run the default analysis with crossing fractures
python main.py
```

This will:
1. Generate a test fracture dataset
2. Create a spatial grid
3. Detect all fracture intersections
4. Generate 7 PNG visualizations
5. Output a detailed connection report to `output/fracture_connections_report.txt`

## Architecture

### Core Modules

```
ResFracPython/
├── geometry.py           # 2D/3D geometric primitives (Point2D, Point3D, BoundingBox)
├── fracture.py          # FractureElement class - represents a vertical rectangular fracture
├── grid.py              # Grid generation and spatial partitioning
├── intersection.py      # Line-rectangle intersection utilities
├── connections.py       # Fracture connection detection (main algorithm)
├── plotting.py          # All visualization functions
├── main.py              # Orchestration and data flow
└── testing/             # Test datasets and utilities
    ├── test_data.py     # Deterministic test fracture generators
    └── random_fractures.py  # Random fracture generation
```

### Algorithm Overview

#### 1. **Spatial Grid Creation**
- Computes domain bounds from fracture centers
- Creates square grid with cell size ≈ 1/4 of fracture dimensions
- Uses sparse matrix to store fracture-cell intersections

#### 2. **Connection Detection** (3-stage process)
```
For each grid cell with fractures:
  ├─ Check fractures within the same cell
  └─ Check fractures in 8 neighboring cells
      └─ For each pair:
          ├─ Fast: Check z-axis range overlap (reject 99% here)
          ├─ Medium: Check 2D line segment intersection (cross product method)
          └─ Medium: Calculate exact 3D intersection point
```

#### 3. **Line Segment Intersection** (Cross Product Method)
- Uses orientation-based detection
- Handles general case (non-collinear intersection)
- Handles special cases (collinear, endpoint touching)

#### 4. **Intersection Coordinate Calculation**
- Solves parametric line equations: `P = P1 + t(P2-P1)`
- Averages z-coordinates from both fractures

### Data Structures

**Sparse Grid Matrix** (efficient storage):
```python
grid_sparse_csr  # Shape: (num_cells, num_fractures)
                 # Non-zero at (cell_i, fracture_j) if fracture j intersects cell i
```

**Connections** (flexible):
```python
connections[i] = [j, k, ...]  # Fracture i intersects fractures j, k, ...
intersection_points[(i,j)] = Point3D(x, y, z)  # Where i and j intersect
```

## Usage Examples

### Basic Usage

```python
from testing.test_data import crossing_fractures
from grid import initialize_grid
from connections import find_fracture_connections
from plotting import generate_connection_report

# Load test data
centers, thetas, length, height = crossing_fractures()

# Initialize grid
grid_sparse_csr, nf, fractures, x_min, x_max, y_min, y_max, grid_size = initialize_grid(
    centers, thetas, length, height
)

# Find connections
connections, intersection_points = find_fracture_connections(
    grid_sparse_csr, fractures, grid_size
)

# Generate report
report = generate_connection_report(
    connections=connections,
    intersection_points=intersection_points,
    filename="report.txt"
)
print(report)
```

### Using Different Datasets

```python
from testing.test_data import vertical_separation_fractures, intersecting_vertical_fractures

# Test with vertically separated fractures (should have fewer connections)
centers, thetas, length, height = vertical_separation_fractures()

# Test with 3D crossing fractures
centers, thetas, length, height = intersecting_vertical_fractures()
```

### Custom Fracture Data

```python
import numpy as np

# Define your own fractures
centers = np.array([
    0, 0, 0,      # Fracture 0: center at origin
    50, 50, 0,    # Fracture 1: center at (50, 50, 0)
    -50, 50, 100, # Fracture 2: center at (-50, 50, 100)
])
thetas = np.array([0, 45, 90])  # Orientations in degrees
length = 100.0
height = 50.0
```

## Output

Running the analysis generates:

### Visualizations (in `output/` directory)
1. **fractures_topdown.png** - Top-down 2D view of fracture centerlines
2. **fractures_topdown_labeled.png** - Same with fracture ID labels
3. **grid_fracture_counts.png** - Grid cells colored by number of intersecting fractures
4. **grid_fracture_counts_labeled.png** - Grid with multi-fracture cell labels
5. **connection_matrix.png** - Heatmap showing fracture-fracture connections
6. **fractures_3d.png** - 3D wireframe perspective
7. **fractures_3d_connections.png** - 3D with connection lines highlighted

### Report (text)
**fracture_connections_report.txt** - Detailed summary including:
- Total fracture count and connection statistics
- Fractures sorted by number of connections
- List of all connections with 3D intersection coordinates

Example:
```
======================================================================
FRACTURE CONNECTION REPORT
======================================================================

SUMMARY STATISTICS
Total number of fractures: 4
Total unique connections: 4
Connected fractures: 4 (100.0%)

DETAILED CONNECTIONS BY FRACTURE
Fracture    0 ↔ Fracture    2: (     -5.00,      -5.00,       0.00)
Fracture    0 ↔ Fracture    3: (     -5.00,       5.00,       0.00)
```

## Performance Characteristics

- **Grid size:** Proportional to domain size / fracture dimensions
- **Sparse matrix:** Typically 95%+ sparse (depends on fracture density)
- **Connection detection:** O(n) grid cells × O(f²) fractures per cell
- **Tested successfully with:** 100+ fractures

## Testing

The project includes several test datasets:

```python
from testing.test_data import (
    crossing_fractures,           # 4 fractures at same depth
    vertical_separation_fractures,  # 6 fractures (3 shallow, 3 deep)
    intersecting_vertical_fractures # 4 fractures in 3D configuration
)
```

Run tests:
```bash
python test_vertical_datasets.py
python test_plotting.py
```

## API Reference

### FractureElement
```python
class FractureElement:
    def __init__(center: Point3D, length: float, height: float, 
                 orientation: float, id: int = None)
    
    def intersects(other: FractureElement) -> bool
    def get_intersection_point(other: FractureElement) -> Optional[Point3D]
    def get_z_range() -> Tuple[float, float]
    def get_bounding_box() -> BoundingBox2D
```

### Grid Module
```python
def initialize_grid(centers, thetas, length, height, 
                   padding_factor=1.0, cell_size_factor=0.25) 
    -> (sparse_matrix, num_fractures, fractures, x_min, x_max, y_min, y_max, grid_size)
```

### Connection Module
```python
def find_fracture_connections(grid_sparse_csr, fractures, grid_size) 
    -> (connections: List[List[int]], intersection_points: Dict)
```

### Plotting Module
```python
def plot_fractures_topdown(centers, thetas, length, height, ...)
def plot_grid_with_fracture_counts(grid_sparse_csr, ...)
def plot_connection_matrix(connections, ...)
def plot_fractures_3d(centers, thetas, length, height, ...)
def generate_connection_report(connections, intersection_points, filename)
```

## Future Enhancements

- [ ] Support for non-vertical fractures
- [ ] Export to VTK format for 3D visualization in ParaView
- [ ] Parallel processing for large fracture networks
- [ ] Web-based interactive visualization
- [ ] Fracture network permeability analysis
- [ ] Unit tests with pytest

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing style and conventions
- All functions have docstrings
- Type hints are included
- Changes maintain backward compatibility

## License

[Add appropriate license]

## Author

[Your name/organization]

## References

- Cross product line intersection method: [Computational Geometry reference]
- Spatial grid indexing: Standard divide-and-conquer approach in computational geometry

