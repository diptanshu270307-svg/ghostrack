"""Sensor simulator package for InSight AI / GhostTrack."""

from simulator.geometry import (
    ccw,
    segments_intersect,
    point_segment_distance,
    check_line_of_sight,
    resolve_wall_collision,
)
from simulator.trajectory import TrajectoryGenerator
from simulator.sim import SensorSimulator, get_default_layout

__all__ = [
    "ccw",
    "segments_intersect",
    "point_segment_distance",
    "check_line_of_sight",
    "resolve_wall_collision",
    "TrajectoryGenerator",
    "SensorSimulator",
    "get_default_layout",
]
