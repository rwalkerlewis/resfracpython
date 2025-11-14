"""
Intersection detection utilities for geometric primitives.
"""

import numpy as np
from .geometry import Point2D
from .fracture import FractureElement


def z_range_overlap(frac1: FractureElement, frac2: FractureElement) -> bool:
    """
    Check if two fractures have overlapping z-axis bounding boxes.
    
    Parameters
    ----------
    frac1 : FractureElement
        First fracture
    frac2 : FractureElement
        Second fracture
    
    Returns
    -------
    bool
        True if the z ranges overlap, False otherwise.
    """
    z_min1, z_max1 = frac1.get_z_range()
    z_min2, z_max2 = frac2.get_z_range()
    
    # Check for overlap: ranges overlap if max1 >= min2 and max2 >= min1
    return z_max1 >= z_min2 and z_max2 >= z_min1


def line_segment_intersects_rect(p1: Point2D, p2: Point2D, rect_x_min: float, rect_x_max: float, 
                                   rect_y_min: float, rect_y_max: float) -> bool:
    """
    Check if a line segment (from p1 to p2) intersects an axis-aligned rectangle.
    Uses parametric line representation and bounds checking.
    
    Args:
        p1, p2: Endpoints of the line segment (Point2D objects)
        rect_x_min, rect_x_max: X bounds of the rectangle
        rect_y_min, rect_y_max: Y bounds of the rectangle
    
    Returns:
        True if the line segment intersects the rectangle, False otherwise
    """
    # Check if either endpoint is inside the rectangle
    if (rect_x_min <= p1.x <= rect_x_max and rect_y_min <= p1.y <= rect_y_max):
        return True
    if (rect_x_min <= p2.x <= rect_x_max and rect_y_min <= p2.y <= rect_y_max):
        return True
    
    # Check if line segment intersects any of the four rectangle edges
    # Use parametric form: point = p1 + t * (p2 - p1), where 0 <= t <= 1
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    
    # Check intersection with vertical edges (x = x_min and x = x_max)
    if abs(dx) > 1e-10:
        for x_val in [rect_x_min, rect_x_max]:
            t = (x_val - p1.x) / dx
            if 0 <= t <= 1:
                y = p1.y + t * dy
                if rect_y_min <= y <= rect_y_max:
                    return True
    
    # Check intersection with horizontal edges (y = y_min and y = y_max)
    if abs(dy) > 1e-10:
        for y_val in [rect_y_min, rect_y_max]:
            t = (y_val - p1.y) / dy
            if 0 <= t <= 1:
                x = p1.x + t * dx
                if rect_x_min <= x <= rect_x_max:
                    return True
    
    return False


def vectorized_line_rect_intersect(p1: Point2D, p2: Point2D,
                                   cell_x_min: np.ndarray, cell_x_max: np.ndarray,
                                   cell_y_min: np.ndarray, cell_y_max: np.ndarray) -> np.ndarray:
    """
    Vectorized line-rectangle intersection check for multiple cells simultaneously.
    
    Args:
        p1, p2: Line segment endpoints (Point2D objects)
        cell_x_min, cell_x_max: Arrays of shape (grid_size, grid_size) with x bounds
        cell_y_min, cell_y_max: Arrays of shape (grid_size, grid_size) with y bounds
    
    Returns:
        Boolean array of shape (grid_size, grid_size) indicating intersections
    """
    result = np.zeros_like(cell_x_min, dtype=bool)
    
    # Check if either endpoint is inside any cell
    p1_inside = (cell_x_min <= p1.x) & (p1.x <= cell_x_max) & \
                (cell_y_min <= p1.y) & (p1.y <= cell_y_max)
    p2_inside = (cell_x_min <= p2.x) & (p2.x <= cell_x_max) & \
                (cell_y_min <= p2.y) & (p2.y <= cell_y_max)
    result = p1_inside | p2_inside
    
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    
    # Check line-rectangle intersection by testing intersection with rectangle edges
    # For each cell, test if line segment intersects the rectangle defined by cell bounds
    
    # Vectorized check for vertical edges (x = cell_x_min and x = cell_x_max)
    if abs(dx) > 1e-10:
        # Check left edge (x = cell_x_min)
        t_left = (cell_x_min - p1.x) / dx
        valid_t_left = (t_left >= 0) & (t_left <= 1)
        y_left = p1.y + t_left * dy
        on_left_edge = valid_t_left & (cell_y_min <= y_left) & (y_left <= cell_y_max)
        result = result | on_left_edge
        
        # Check right edge (x = cell_x_max)
        t_right = (cell_x_max - p1.x) / dx
        valid_t_right = (t_right >= 0) & (t_right <= 1)
        y_right = p1.y + t_right * dy
        on_right_edge = valid_t_right & (cell_y_min <= y_right) & (y_right <= cell_y_max)
        result = result | on_right_edge
    
    # Vectorized check for horizontal edges (y = cell_y_min and y = cell_y_max)
    if abs(dy) > 1e-10:
        # Check bottom edge (y = cell_y_min)
        t_bottom = (cell_y_min - p1.y) / dy
        valid_t_bottom = (t_bottom >= 0) & (t_bottom <= 1)
        x_bottom = p1.x + t_bottom * dx
        on_bottom_edge = valid_t_bottom & (cell_x_min <= x_bottom) & (x_bottom <= cell_x_max)
        result = result | on_bottom_edge
        
        # Check top edge (y = cell_y_max)
        t_top = (cell_y_max - p1.y) / dy
        valid_t_top = (t_top >= 0) & (t_top <= 1)
        x_top = p1.x + t_top * dx
        on_top_edge = valid_t_top & (cell_x_min <= x_top) & (x_top <= cell_x_max)
        result = result | on_top_edge
    
    return result
