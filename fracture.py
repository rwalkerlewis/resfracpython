"""
Fracture element representation
"""

import numpy as np
from geometry import Point2D, Point3D, BoundingBox2D, degrees_to_radians


class FractureElement:
    """
    Represents a vertical rectangular fracture element.
    
    The fracture is a vertical rectangle with:
    - Center at (x, y, z)
    - Length L and height H
    - Orientation theta (clockwise from positive y-axis)
    """

    def __init__(self, center: Point3D, length: float, height: float, orientation: float):
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
        return (f"FractureElement(center={self.center}, "
                f"L={self.length}, H={self.height}, "
                f"theta={self.orientation}°)")