# Project Structure

## Overview

ResFracPython is organized into logical modules following separation of concerns:

## Core Modules

### `geometry.py`
**Purpose**: Geometric primitives and utilities
**Classes**:
- `Point2D` - 2D point with (x, y) coordinates
- `Point3D` - 3D point with (x, y, z) coordinates
- `BoundingBox2D` - 2D bounding box for fracture projections

**Key Functions**:
- `degrees_to_radians()` - Angle conversion utility

### `fracture.py`
**Purpose**: Represents individual fractures and intersection detection
**Classes**:
- `FractureElement` - A vertical rectangular fracture
  - Properties: center position, length, height, orientation, depth range
  - Methods: intersection testing, bounding box, endpoints

**Key Methods**:
- `intersects(other)` - Check if two fractures intersect (with z-axis screening)
- `get_intersection_point(other)` - Get 3D coordinates where fractures intersect
- `get_z_range()` - Get depth range of fracture
- `get_bounding_box()` - Get 2D bounding box
- `get_line_segment_endpoints()` - Get centerline endpoints for intersection testing
- `_line_segments_intersect()` - Static method using cross-product algorithm

**Key Function**:
- `create_fracture_elements()` - Factory function to create FractureElement objects from raw data

### `grid.py`
**Purpose**: Spatial grid generation and management
**Key Functions**:
- `compute_grid_bounds()` - Calculate domain bounds with padding
- `compute_grid_size()` - Determine grid dimensions
- `create_grid()` - Build sparse matrix of fracture-cell intersections
- `initialize_grid()` - Main entry point for grid creation

**Returns**:
- `grid_sparse_csr` - Sparse matrix where (cell, fracture) indicates intersection
- `fractures` - List of FractureElement objects
- Grid bounds and dimensions

### `intersection.py`
**Purpose**: Line-rectangle intersection utilities
**Key Functions**:
- `z_range_overlap()` - Check if two fractures' z-ranges overlap (fast screening)
- `line_segment_intersects_rect()` - Check if a line segment intersects an axis-aligned rectangle
- `vectorized_line_rect_intersect()` - Vectorized version for checking many cells at once

### `connections.py`
**Purpose**: Main algorithm for detecting fracture connections
**Key Function**:
- `find_fracture_connections()` - Finds all fracture intersections
  - Iterates through grid cells with fractures
  - Checks within-cell and cross-neighbor pairs
  - Uses hierarchical screening: z-axis overlap → line intersection → coordinate calculation
  - Returns connections list and intersection point dictionary

**Returns**:
- `connections[i]` - List of fracture indices that intersect with fracture i
- `intersection_points[(i,j)]` - Point3D coordinates where fractures i and j intersect

### `plotting.py`
**Purpose**: All visualization and reporting functions
**Key Functions**:
- `plot_fractures_topdown()` - 2D top-down view of fractures
- `plot_grid_with_fracture_counts()` - Grid cells colored by fracture count
- `plot_connection_matrix()` - Heatmap of fracture-fracture connections
- `plot_fractures_3d()` - 3D perspective wireframe visualization
- `generate_connection_report()` - Text report of all connections with coordinates

**Features**:
- Multiple visualization modes
- Automatic scaling and labeling
- Support for large grids (adaptive label density)

### `main.py`
**Purpose**: Orchestration and data flow
**Workflow**:
1. Load/generate fracture data
2. Initialize grid
3. Plot initial visualizations
4. Find connections
5. Generate report and visualizations

**Output**:
- 7 PNG files in `output/` directory
- Text report with connection details

## Testing & Examples

### `testing/`
Directory containing test data generators and utilities

#### `testing/test_data.py`
**Purpose**: Deterministic test fracture datasets
**Functions**:
- `crossing_fractures()` - 4 fractures intersecting in a grid pattern (all at same depth)
- `vertical_separation_fractures()` - 6 fractures in two depth groups (tests z-axis screening)
- `intersecting_vertical_fractures()` - 4 fractures with significant z-differences (tests 3D)

#### `testing/random_fractures.py`
**Purpose**: Random fracture generation for testing scalability
**Functions**:
- `generate_fracture_data()` - Creates random fracture dataset

### `test_plotting.py`
**Purpose**: Test visualizations with a specific dataset
**Usage**: `python test_plotting.py`

### `test_vertical_datasets.py`
**Purpose**: Validate z-axis screening functionality
**Usage**: `python test_vertical_datasets.py`

### `demo.py`
**Purpose**: Comprehensive demo script analyzing multiple datasets
**Usage**: `python demo.py`
**Output**: Creates `output/` subdirectories for each dataset

## Configuration Files

### `README.md`
- Project overview
- Installation instructions
- Quick start guide
- Architecture documentation
- API reference
- Usage examples

### `setup.py`
- Package metadata
- Dependency specifications
- Installation configuration

### `requirements.txt`
- Simple list of dependencies with version constraints

### `.gitignore`
- Standard Python ignore patterns
- IDE configuration files
- Output directories

### `CONTRIBUTING.md`
- Code style guidelines
- Documentation requirements
- Contribution process
- Areas for enhancement

## Directory Structure

```
resfracpython/
├── README.md              # Project documentation
├── setup.py              # Package configuration
├── requirements.txt      # Dependency list
├── CONTRIBUTING.md       # Contribution guidelines
├── .gitignore           # Git ignore patterns
│
├── Core Modules:
├── geometry.py          # Geometric primitives
├── fracture.py          # Fracture representation
├── grid.py              # Spatial grid
├── intersection.py      # Intersection utilities
├── connections.py       # Connection detection (main algorithm)
├── plotting.py          # Visualization
├── main.py              # Orchestration
│
├── Examples & Tests:
├── demo.py              # Comprehensive demo
├── test_plotting.py     # Visualization test
├── test_vertical_datasets.py  # Z-axis screening test
│
├── testing/             # Test utilities
│   ├── test_data.py     # Test fracture datasets
│   └── random_fractures.py   # Random generation
│
└── output/              # Generated visualizations (created at runtime)
    ├── *.png            # Visualization files
    └── *.txt            # Connection reports
```

## Data Flow

```
Input Data (centers, thetas, length, height)
    ↓
fracture.py: Create FractureElement objects
    ↓
grid.py: Initialize spatial grid
    ↓
connections.py: Detect fracture intersections
    ├─ For each grid cell:
    ├─ Check within-cell pairs
    └─ Check cross-neighbor pairs
    ↓
plotting.py: Generate visualizations & reports
    ↓
output/ directory: Save results
```

## Typical Development Workflow

1. **Add new geometry feature** → Edit `geometry.py`
2. **Change fracture representation** → Edit `fracture.py`
3. **Optimize grid creation** → Edit `grid.py`
4. **Improve connection detection** → Edit `connections.py`
5. **Add visualization** → Edit `plotting.py`
6. **Test changes** → Use `testing/test_data.py` datasets
7. **Run full analysis** → Execute `demo.py` or `main.py`

## Key Design Patterns

1. **Separation of Concerns**: Each module has a single responsibility
2. **Factory Pattern**: `create_fracture_elements()` creates objects from raw data
3. **Vectorization**: NumPy operations for performance
4. **Sparse Matrices**: Efficient storage of grid data
5. **Hierarchical Screening**: Fast rejection tests before expensive computations
