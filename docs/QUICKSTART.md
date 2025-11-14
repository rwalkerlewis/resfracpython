# Quick Reference Guide

## Installation

```bash
git clone https://github.com/rwalkerlewis/resfracpython.git
cd resfracpython
pip install -r requirements.txt
```

## Running the Analysis

### Option 1: Run Default Analysis
```bash
python main.py
```
Analyzes 4 crossing fractures and generates visualizations and report in `output/`.

### Option 2: Run Comprehensive Demo
```bash
python demo.py
```
Analyzes 3 different test datasets and creates organized output in `output/[dataset_name]/`.

### Option 3: Custom Analysis (Code)
```python
from grid import initialize_grid
from connections import find_fracture_connections
from plotting import generate_connection_report
import numpy as np

# Define your fractures
centers = np.array([0, 0, 0, 100, 100, 0])  # Two fractures
thetas = np.array([0, 45])                   # Orientations
length = 200.0
height = 50.0

# Run analysis
grid, nf, fractures, x_min, x_max, y_min, y_max, grid_size = initialize_grid(
    centers, thetas, length, height
)

connections, intersection_points = find_fracture_connections(
    grid, fractures, grid_size
)

# Generate report
report = generate_connection_report(connections, intersection_points)
print(report)
```

## Data Input Format

### Centers Array
Flat numpy array with shape `(nf*3,)`:
```python
centers = np.array([
    x0, y0, z0,    # Fracture 0 center
    x1, y1, z1,    # Fracture 1 center
    ...
])
```

### Thetas Array
Orientation angles in degrees (clockwise from +y axis):
```python
thetas = np.array([0, 45, 90, 135])  # 4 fractures at different angles
```

### Geometry Parameters
- `length` - Fracture length (horizontal extent)
- `height` - Fracture height (vertical extent)

## Output Files

### Visualizations (PNG)
Generated in `output/` directory:
- `01_topdown.png` - 2D view of fracture centerlines
- `02_topdown_labeled.png` - With fracture ID labels
- `03_grid.png` - Grid cells colored by fracture count
- `04_grid_labeled.png` - Grid with multi-fracture cell labels
- `05_connection_matrix.png` - Heatmap of connections
- `06_3d_view.png` - 3D wireframe perspective
- `07_3d_connections.png` - 3D with connection lines

### Reports (TXT)
- `connection_report.txt` - Detailed connection analysis with coordinates

## API Quick Reference

### FractureElement
```python
from fracture import FractureElement

f = FractureElement(center=Point3D(0, 0, 0), length=200, height=50, 
                   orientation=45, id=0)

# Check intersection
if f.intersects(other_fracture):
    print("Fractures intersect!")

# Get intersection point
pt = f.get_intersection_point(other_fracture)
print(f"Intersection at: ({pt.x}, {pt.y}, {pt.z})")

# Get z-range
z_min, z_max = f.get_z_range()
```

### Grid Module
```python
from grid import initialize_grid

grid_csr, nf, fractures, x_min, x_max, y_min, y_max, grid_size = initialize_grid(
    centers, thetas, length, height,
    padding_factor=1.0,           # Add padding around fractures
    cell_size_factor=0.25         # Cell size = 0.25 * max(length, height)
)

# Access grid properties
num_cells = grid_size * grid_size
fractures_in_cell_0 = grid_csr[0, :].nonzero()[1]  # Fracture indices in cell 0
```

### Connection Module
```python
from connections import find_fracture_connections

connections, intersection_points = find_fracture_connections(
    grid_sparse_csr=grid_csr,
    fractures=fractures,
    grid_size=grid_size
)

# Access results
print(f"Fracture 0 connects to: {connections[0]}")

for (i, j), point in intersection_points.items():
    print(f"F{i} and F{j} intersect at ({point.x}, {point.y}, {point.z})")
```

### Plotting Module
```python
from plotting import plot_fractures_topdown, plot_fractures_3d, generate_connection_report

# Plot top-down view
plot_fractures_topdown(centers, thetas, length, height, 
                      show_labels=True, filename="my_plot.png")

# Plot 3D view
plot_fractures_3d(centers, thetas, length, height,
                 connections=connections, show_labels=True,
                 filename="my_3d_plot.png")

# Generate text report
report = generate_connection_report(connections, intersection_points, 
                                  filename="my_report.txt")
```

## Test Datasets

### Using Built-in Datasets
```python
from testing.test_data import (
    crossing_fractures,           # 4 fractures, all same depth
    vertical_separation_fractures, # 6 fractures, 2 depth groups
    intersecting_vertical_fractures # 4 fractures in 3D
)

centers, thetas, length, height = crossing_fractures()
# ... continue with analysis
```

### Generating Random Fractures
```python
from testing.random_fractures import generate_fracture_data

centers, thetas = generate_fracture_data(
    num_fractures=50,
    seed=42
)
length = 200.0
height = 50.0
```

## Performance Tips

1. **Grid Size**: Automatically computed based on fracture dimensions
   - Smaller grid = faster but less granular
   - Adjust via `cell_size_factor` in `initialize_grid()`

2. **Z-Axis Screening**: Dramatically reduces expensive tests
   - ~99% of pairs rejected without full intersection test
   - Critical for performance with large networks

3. **Sparse Matrices**: Memory efficient
   - Typical grids are 90%+ sparse
   - Only non-zero entries stored

## Common Issues

### "Grid too fine/coarse"
Adjust `cell_size_factor` in `initialize_grid()`:
```python
grid, ... = initialize_grid(
    centers, thetas, length, height,
    cell_size_factor=0.5  # Larger cells = coarser grid
)
```

### Missing connections between distant fractures
Ensure `padding_factor` is large enough in `initialize_grid()`:
```python
grid, ... = initialize_grid(
    centers, thetas, length, height,
    padding_factor=2.0  # Increase padding
)
```

### Memory issues with large datasets
- Use sparse matrix operations
- Ensure grid size isn't too fine
- Consider processing in batches

## File Organization

```
resfracpython/
├── Core modules: geometry.py, fracture.py, grid.py, intersection.py,
│                 connections.py, plotting.py
├── Entry points: main.py, demo.py
├── Tests:        testing/test_data.py, test_plotting.py,
│                 test_vertical_datasets.py
├── Docs:         README.md, PROJECT_STRUCTURE.md, CONTRIBUTING.md
├── Config:       requirements.txt, setup.py, .gitignore
└── Output:       output/  (created at runtime)
```

## Next Steps

1. **Explore test datasets**: Run `python demo.py`
2. **Study the code**: Start with `main.py`, then `connections.py`
3. **Try custom data**: Create your own fracture dataset
4. **Extend functionality**: Add features following CONTRIBUTING.md guidelines

## Support

For issues or questions:
1. Check README.md for detailed documentation
2. Open an issue on GitHub with clear description
