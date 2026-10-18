"""
Map-Aware Multi-Modal 2D Belief Heatmap Generator (USPs #01, #04, #07).
Computes a normalized 2D probability density matrix over the room grid:
  - Supports multi-modal plausible locations (Ambiguous bimodal corridor belief).
  - Applies Map-Aware wall masking so solid walls cannot hold target probability mass.
  - Collapses smoothly from a wide blob (uncertain/Ghost) to a sharp peak (Tracked).
"""

from typing import List, Tuple
import numpy as np

from shared.schemas import Point2D, WallSegment


def compute_belief_grid(
    x: float,
    y: float,
    sigma: float,
    room_width: float = 12.0,
    room_height: float = 10.0,
    grid_rows: int = 30,
    grid_cols: int = 36,
    plausible_locations: List[Point2D] | None = None,
    walls: List[WallSegment] | None = None,
) -> Tuple[List[List[float]], Tuple[int, int]]:
    """
    Build a 2D normalized belief probability grid of shape (grid_rows, grid_cols).
    Row 0 corresponds to y near 0, Row (grid_rows-1) to y near room_height.
    """
    xs = np.linspace(0.0, room_width, grid_cols)
    ys = np.linspace(0.0, room_height, grid_rows)
    grid_x, grid_y = np.meshgrid(xs, ys)

    eff_sigma = max(0.22, float(sigma))
    two_sigma_sq = 2.0 * eff_sigma * eff_sigma

    modes = plausible_locations if plausible_locations else [Point2D(x=x, y=y, weight=1.0)]
    belief = np.zeros((grid_rows, grid_cols), dtype=float)

    for mode in modes:
        dist_sq = (grid_x - mode.x) ** 2 + (grid_y - mode.y) ** 2
        belief += mode.weight * np.exp(-dist_sq / two_sigma_sq)

    # Map-Aware Wall Masking (USP #07): zero out cells intersecting interior walls
    if walls:
        for w in walls:
            dx = w.x2 - w.x1
            dy = w.y2 - w.y1
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq > 1e-6:
                t = np.clip(
                    ((grid_x - w.x1) * dx + (grid_y - w.y1) * dy) / seg_len_sq,
                    0.0,
                    1.0,
                )
                proj_x = w.x1 + t * dx
                proj_y = w.y1 + t * dy
                dist_to_wall = np.hypot(grid_x - proj_x, grid_y - proj_y)
                wall_mask = np.where(dist_to_wall < 0.22, 0.08, 1.0)
                belief *= wall_mask

    max_val = float(np.max(belief))
    if max_val > 1e-9:
        belief = belief / max_val

    rounded = np.round(belief, 4).tolist()
    return rounded, (grid_rows, grid_cols)
