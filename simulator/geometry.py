"""
Geometric utility functions for the indoor positioning simulator.
Includes robust 2D line segment intersection tests, distance queries,
and boundary collision handling.
"""

from typing import Tuple, List, Optional
import math
from shared.schemas import WallSegment


def ccw(A: Tuple[float, float], B: Tuple[float, float], C: Tuple[float, float]) -> bool:
    """
    Test whether three points are listed in counterclockwise order.
    Uses the 2D cross product determinant (B - A) x (C - A).
    """
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def segments_intersect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float],
) -> bool:
    """
    Determine if line segment p1-p2 intersects line segment p3-p4.
    Handles proper crossings and collinear bounding box overlaps.
    """
    # Quick rejection with axis-aligned bounding boxes
    if max(p1[0], p2[0]) < min(p3[0], p4[0]) or min(p1[0], p2[0]) > max(p3[0], p4[0]):
        return False
    if max(p1[1], p2[1]) < min(p3[1], p4[1]) or min(p1[1], p2[1]) > max(p3[1], p4[1]):
        return False

    return (ccw(p1, p3, p4) != ccw(p2, p3, p4)) and (ccw(p1, p2, p3) != ccw(p1, p2, p4))


def point_segment_distance(
    p: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
) -> float:
    """
    Compute shortest Euclidean distance from point p to segment a-b.
    """
    px, py = p
    ax, ay = a
    bx, by = b

    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq < 1e-9:
        return math.hypot(px - ax, py - ay)

    # Project p onto ab, clamped between [0, 1]
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
    proj_x = ax + t * dx
    proj_y = ay + t * dy

    return math.hypot(px - proj_x, py - proj_y)


def check_line_of_sight(
    p_target: Tuple[float, float],
    p_anchor: Tuple[float, float],
    walls: List[WallSegment],
) -> Tuple[bool, int]:
    """
    Check if the straight line path between target and anchor is obstructed by any wall.
    
    Returns:
        (has_los, wall_cut_count):
            has_los is True if 0 walls block line of sight.
            wall_cut_count is total number of wall segments penetrated.
    """
    obstructions = 0
    for wall in walls:
        w_start = (wall.x1, wall.y1)
        w_end = (wall.x2, wall.y2)
        if segments_intersect(p_target, p_anchor, w_start, w_end):
            obstructions += 1

    return (obstructions == 0, obstructions)


def resolve_wall_collision(
    pos: Tuple[float, float],
    velocity: Tuple[float, float],
    walls: List[WallSegment],
    room_width: float,
    room_height: float,
    margin: float = 0.35,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Enforce soft boundary and interior wall collision reflections.
    Keeps position within bounds and mirrors velocity on impact.
    """
    x, y = pos
    vx, vy = velocity

    # Room exterior boundaries
    if x < margin:
        x = margin
        vx = abs(vx)
    elif x > room_width - margin:
        x = room_width - margin
        vx = -abs(vx)

    if y < margin:
        y = margin
        vy = abs(vy)
    elif y > room_height - margin:
        y = room_height - margin
        vy = -abs(vy)

    # Interior walls collision check
    for wall in walls:
        dist = point_segment_distance((x, y), (wall.x1, wall.y1), (wall.x2, wall.y2))
        if dist < margin:
            # Wall normal vector
            dx = wall.x2 - wall.x1
            dy = wall.y2 - wall.y1
            wall_len = math.hypot(dx, dy)
            if wall_len > 1e-6:
                nx = -dy / wall_len
                ny = dx / wall_len
                # Push out along normal
                dot = vx * nx + vy * ny
                vx = vx - 2 * dot * nx
                vy = vy - 2 * dot * ny
                x += nx * (margin - dist) * 1.1
                y += ny * (margin - dist) * 1.1

    return (x, y), (vx, vy)
