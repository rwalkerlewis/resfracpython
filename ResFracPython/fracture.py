"""
Fracture element representation
"""

import numpy as np
from typing import Optional, List
from .geometry import Point2D, Point3D, BoundingBox2D, degrees_to_radians


class FractureElement:
    """
    Represents a vertical rectangular fracture element.
    
    The fracture is a vertical rectangle with:
    - Center at (x, y, z)
    - Length L and height H
    - Orientation theta (clockwise from positive y-axis)
    """

    def __init__(self, center: Point3D, length: float, height: float, orientation: float, id: Optional[int] = None):
        """
        Initialize a fracture element.
        
        Args:
            center: 3D center point (x, y, z)
            length: Length of the fracture in the x-y plane
            height: Height of the fracture in the z direction
            orientation: Orientation angle in degrees (clockwise from positive y-axis)
        """
        self.center = center
        self.length = length
        self.height = height
        self.orientation = orientation
        # Assign an identifier (based on original data order) if provided
        self.id = id

        # Create 2D bounding box projection using the same angle convention as input
        # Fractures have negligible thickness, so width is essentially 0
        # We use a very small width for numerical stability in bounding box calculations
        # For actual intersection, we use line segment intersection
        self.xy_bounding_box = BoundingBox2D(
            center.to_2d(),
            length,
            1e-10,  # Negligible width
            orientation  # Same convention: clockwise from +y axis
        )

        # Store angle in radians for line segment calculations
        # Convert orientation (clockwise from +y) to radians
        self.angle_rad = np.pi / 2.0 - degrees_to_radians(orientation)

    def get_z_range(self) -> tuple:
        """Get the z-coordinate range [z_min, z_max]."""
        half_height = self.height / 2.0
        return (self.center.z - half_height, self.center.z + half_height)
    
    def get_line_segment_endpoints(self) -> tuple:
        """
        Get the endpoints of the line segment representing this fracture in x-y plane.
        Returns (p1, p2) where p1 and p2 are Point2D.
        Uses the same angle convention as input (clockwise from positive y-axis).
        """
        half_length = self.length / 2.0
        # Direction vector (unit vector along the fracture)
        # angle_rad is already converted from orientation (clockwise from +y)
        cos_a = np.cos(self.angle_rad)
        sin_a = np.sin(self.angle_rad)
        dx = cos_a * half_length
        dy = sin_a * half_length
        
        center_2d = self.center.to_2d()
        p1 = Point2D(center_2d.x - dx, center_2d.y - dy)
        p2 = Point2D(center_2d.x + dx, center_2d.y + dy)
        return (p1, p2)
    
    def get_bounding_box(self) -> BoundingBox2D:
        """
        Get the 2D bounding box projection of this fracture.
        Returns a BoundingBox2D using the same angle convention as input.
        """
        return self.xy_bounding_box
    
    def z_overlaps(self, other: 'FractureElement') -> bool:
        """Check if z-ranges overlap."""
        z_min1, z_max1 = self.get_z_range()
        z_min2, z_max2 = other.get_z_range()
        return not (z_max1 < z_min2 or z_max2 < z_min1)
    
    @staticmethod
    def _line_segments_intersect(p1: Point2D, p2: Point2D, q1: Point2D, q2: Point2D) -> bool:
        """
        Check if two line segments intersect using cross product method.
        Line segment 1: from p1 to p2
        Line segment 2: from q1 to q2
        """
        def cross_product(o: Point2D, a: Point2D, b: Point2D) -> float:
            """Cross product of vectors (a-o) and (b-o)."""
            return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
        
        def on_segment(p: Point2D, q: Point2D, r: Point2D) -> bool:
            """Check if point r lies on segment pq."""
            if (min(p.x, q.x) <= r.x <= max(p.x, q.x) and
                min(p.y, q.y) <= r.y <= max(p.y, q.y)):
                return True
            return False
        
        # Check if segments intersect using orientation method
        o1 = cross_product(p1, p2, q1)
        o2 = cross_product(p1, p2, q2)
        o3 = cross_product(q1, q2, p1)
        o4 = cross_product(q1, q2, p2)
        
        # General case: segments intersect if orientations are different
        if ((o1 > 0 and o2 < 0) or (o1 < 0 and o2 > 0)) and \
           ((o3 > 0 and o4 < 0) or (o3 < 0 and o4 > 0)):
            return True
        
        # Special cases: collinear points
        if o1 == 0 and on_segment(p1, p2, q1):
            return True
        if o2 == 0 and on_segment(p1, p2, q2):
            return True
        if o3 == 0 and on_segment(q1, q2, p1):
            return True
        if o4 == 0 and on_segment(q1, q2, p2):
            return True
        
        return False
    
    def get_intersection_point(self, other: 'FractureElement') -> Optional[Point3D]:
        """
        Get the 3D intersection point of two fractures if they intersect.
        
        Returns None if fractures don't intersect.
        Returns a Point3D with (x, y, z_avg) where z_avg is the average z-coordinate.
        """
        # Check if they intersect first
        if not self.intersects(other):
            return None
        
        # Get line segment endpoints
        p1, p2 = self.get_line_segment_endpoints()
        q1, q2 = other.get_line_segment_endpoints()
        
        # Find intersection point in 2D (xy plane)
        # Using parametric form: intersection point = p1 + t*(p2-p1)
        x1, y1 = p1.x, p1.y
        x2, y2 = p2.x, p2.y
        x3, y3 = q1.x, q1.y
        x4, y4 = q2.x, q2.y
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-10:  # Lines are parallel or coincident
            # Return midpoint of first line as approximation
            return Point3D((x1 + x2) / 2, (y1 + y2) / 2, 
                          (self.center.z + other.center.z) / 2)
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        # Calculate intersection point
        x_int = x1 + t * (x2 - x1)
        y_int = y1 + t * (y2 - y1)
        
        # Z coordinate is average of the two fracture centers (since they're vertical planes)
        z_int = (self.center.z + other.center.z) / 2
        
        return Point3D(x_int, y_int, z_int)
    
    def intersects(self, other: 'FractureElement') -> bool:
        """
        Check if this fracture intersects another fracture.
        
        Two vertical fractures intersect if:
        1. Their z-ranges overlap
        2. Their line segment projections onto the x-y plane intersect
        """
        # Quick rejection: check z-ranges first
        if not self.z_overlaps(other):
            return False
        
        # Get line segment endpoints for both fractures
        p1, p2 = self.get_line_segment_endpoints()
        q1, q2 = other.get_line_segment_endpoints()
        
        # Check if line segments intersect
        return self._line_segments_intersect(p1, p2, q1, q2)
    
    def __repr__(self):
        id_str = f", id={self.id}" if getattr(self, 'id', None) is not None else ""
        return (f"FractureElement(center={self.center}, "
                f"L={self.length}, H={self.height}, "
                f"theta={self.orientation}°{id_str})")


def create_fracture_elements(
    centers: np.ndarray,
    orientations: np.ndarray,
    length: float,
    height: float,
    x_min: float = None,
    x_max: float = None,
    y_min: float = None,
    y_max: float = None,
    grid_size: int = None
) -> tuple:
    """
    Create FractureElement objects and optionally compute grid cell intersections.
    
    Args:
        centers: Flat array of length Nf*3: [x0, y0, z0, x1, y1, z1, ...]
        orientations: Array of length Nf: [theta0, theta1, ...]
        length: Length L of each fracture
        height: Height H of each fracture
        x_min, x_max, y_min, y_max: Grid domain bounds (optional, for intersection computation)
        grid_size: Grid dimensions per side (optional, for intersection computation)
    
    Returns:
        If grid parameters provided:
            Tuple of (fractures, all_corners_x_array, all_corners_y_array, intersections) where:
            - fractures: NumPy array (dtype=object) of FractureElement objects
            - all_corners_x_array: numpy array of shape (nf, 4) with x-coordinates of bounding box corners
            - all_corners_y_array: numpy array of shape (nf, 4) with y-coordinates of bounding box corners
            - intersections: Boolean array of shape (nf, grid_size, grid_size) indicating grid cell intersections
        Otherwise:
            Tuple of (fractures, all_corners_x_array, all_corners_y_array, None)
    """
    nf = len(orientations)
    if len(centers) != nf * 3:
        raise ValueError(f"centers length ({len(centers)}) must be {nf} * 3 = {nf * 3}")
    
    fractures_list = []
    all_corners_x_array = np.zeros((nf, 4), dtype=np.float64)
    all_corners_y_array = np.zeros((nf, 4), dtype=np.float64)
    intersections = None
    
    # Compute grid cell bounds if needed
    compute_intersections = all([x is not None for x in [x_min, x_max, y_min, y_max, grid_size]])
    if compute_intersections:
        from .intersection import vectorized_line_rect_intersect
        
        dx = (x_max - x_min) / grid_size
        dy = (y_max - y_min) / grid_size
        
        # Pre-compute grid cell bounds
        i_indices = np.arange(grid_size)
        j_indices = np.arange(grid_size)
        ii, jj = np.meshgrid(i_indices, j_indices, indexing='ij')
        
        cell_x_min_grid = x_min + ii * dx
        cell_x_max_grid = x_min + (ii + 1) * dx
        cell_y_min_grid = y_min + jj * dy
        cell_y_max_grid = y_min + (jj + 1) * dy
        
        intersections = np.zeros((nf, grid_size, grid_size), dtype=bool)
    
    for i in range(nf):
        x = centers[i * 3]
        y = centers[i * 3 + 1]
        z = centers[i * 3 + 2]
        center = Point3D(x, y, z)
        orientation = orientations[i]
        fracture = FractureElement(center, length, height, orientation, id=i)
        fractures_list.append(fracture)
        
        # Compute and store bounding box corners at the same time
        corners = fracture.get_bounding_box().get_corners()
        for j, corner in enumerate(corners):
            all_corners_x_array[i, j] = corner.x
            all_corners_y_array[i, j] = corner.y
        
        # Compute grid intersections if requested
        if compute_intersections:
            p1, p2 = fracture.get_line_segment_endpoints()
            intersections[i] = vectorized_line_rect_intersect(
                p1, p2,
                cell_x_min_grid, cell_x_max_grid,
                cell_y_min_grid, cell_y_max_grid
            )
    
    # Convert to NumPy array of objects for consistency with array-based approach
    fractures = np.array(fractures_list, dtype=object)
    
    return fractures, all_corners_x_array, all_corners_y_array, intersections