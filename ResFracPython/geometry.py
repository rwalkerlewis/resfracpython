"""
Geometry classes / operations for fracture connection detection.
"""

import numpy as np
from typing import List, Tuple

class Point2D:
    """2D Point (x,y)"""
    
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        
    def __repr__(self):
        return f"Point2D({self.x}, {self.y})"
    
    def __sub__(self, other):
        """Vec subtraction"""
        return Point2D(self.x - other.x, self.y - other.y)
    
    def __add__(self, other):
        """Vec addition"""
        return Point2D(self.x + other.x, self.y + other.y)

    def rotate(self, angle: float) -> 'Point2D':
        """Rotate point counterclockwise by angle (in radians)."""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return Point2D(
            self.x * cos_a - self.y * sin_a,
            self.x * sin_a + self.y * cos_a
        )
    
    def dot(self, other: 'Point2D') -> float:
        """Dot product."""
        return self.x * other.x + self.y * other.y


class Point3D:
    """3D point with x, y, and z coordinates."""
    
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
    
    def __repr__(self):
        return f"Point3D({self.x}, {self.y}, {self.z})"
    
    def to_2d(self) -> Point2D:
        """Project to x-y plane."""
        return Point2D(self.x, self.y)

class BoundingBox2D:
    """
    Represents a 2D bounding box (axis-aligned or rotated).
    Uses angle convention: clockwise from positive y-axis (fracture strike convention).
    """
    
    def __init__(self, center: Point2D, length: float, width: float, orientation: float):
        """
        Initialize a bounding box.
        
        Args:
            center: Center point of the bounding box
            length: Length of the bounding box (along the orientation direction)
            width: Width of the bounding box (perpendicular to length)
            orientation: Orientation angle in degrees (clockwise from positive y-axis)
        """
        self.center = center
        self.length = length
        self.width = width
        self.orientation = orientation  # degrees, clockwise from +y axis
        
        # Convert to radians for calculations
        # Orientation is clockwise from +y axis
        # +y axis is at 90° counterclockwise from +x axis
        # So: angle_rad = pi/2 - orientation_rad (converting clockwise to counterclockwise)
        orientation_rad = degrees_to_radians(orientation)
        self.angle_rad = np.pi / 2.0 - orientation_rad
    
    def get_corners(self) -> List[Point2D]:
        """Get the four corners of the rotated bounding box."""
        half_length = self.length / 2.0
        half_width = self.width / 2.0
        
        # Corners in local coordinate system (before rotation)
        # Local system: length along x-axis, width along y-axis
        local_corners = np.array([
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width]
        ])
        
        # Rotate and translate
        corners = []
        for corner_coords in local_corners:
            corner = Point2D(corner_coords[0], corner_coords[1])
            rotated = corner.rotate(self.angle_rad)
            corners.append(rotated + self.center)
        
        return corners
    
    def get_axes(self) -> List[Point2D]:
        """Get the two normalized axes of the bounding box."""
        cos_a = np.cos(self.angle_rad)
        sin_a = np.sin(self.angle_rad)
        axis1 = Point2D(cos_a, sin_a)  # Length direction
        axis2 = Point2D(-sin_a, cos_a)  # Width direction
        return np.array([axis1, axis2])
    
    def get_projection_range(self, axis: Point2D) -> Tuple[float, float]:
        """
        Project bounding box onto an axis and return the min and max values.
        Used for Separating Axis Theorem (SAT).
        """
        corners = self.get_corners()
        projections = [corner.dot(axis) for corner in corners]
        return (min(projections), max(projections))
    
    def intersects(self, other: 'BoundingBox2D') -> bool:
        """
        Check if this bounding box intersects another using Separating Axis Theorem.
        """
        # Get all axes to test (2 from each bounding box)
        axes = self.get_axes() + other.get_axes()
        
        # Test each axis
        for axis in axes:
            # Normalize axis
            norm = np.sqrt(axis.x**2 + axis.y**2)
            if norm < 1e-10:
                continue
            normalized_axis = Point2D(axis.x / norm, axis.y / norm)
            
            # Project both bounding boxes onto this axis
            min1, max1 = self.get_projection_range(normalized_axis)
            min2, max2 = other.get_projection_range(normalized_axis)
            
            # If projections don't overlap, bounding boxes don't intersect
            if max1 < min2 or max2 < min1:
                return False
        
        # All axes have overlapping projections, bounding boxes intersect
        return True
    
    def contains_point(self, point: Point2D) -> bool:
        """Check if a point is inside this bounding box."""
        # Transform point to local coordinate system
        local_point = point - self.center
        # Rotate point back to local coordinate system
        cos_a = np.cos(-self.angle_rad)
        sin_a = np.sin(-self.angle_rad)
        rotated_point = Point2D(
            local_point.x * cos_a - local_point.y * sin_a,
            local_point.x * sin_a + local_point.y * cos_a
        )
        
        # Check if point is within bounds
        half_length = self.length / 2.0
        half_width = self.width / 2.0
        return (abs(rotated_point.x) <= half_length and 
                abs(rotated_point.y) <= half_width)


def degrees_to_radians(degrees: float) -> float:
    """Convert degrees to radians."""
    return degrees * np.pi / 180.0


def radians_to_degrees(radians: float) -> float:
    """Convert radians to degrees."""
    return radians * 180.0 / np.pi