"""
Unit tests for the FastAPI Backend pipeline service and Map-Aware Belief Heatmap generator.
"""

from shared.schemas import Point2D, WallSegment
from backend.belief import compute_belief_grid
from backend.main import GhostTrackPipelineService


def test_compute_belief_grid_unimodal_and_bimodal():
    # Unimodal belief grid
    grid, shape = compute_belief_grid(x=6.0, y=5.0, sigma=0.5, room_width=12.0, room_height=10.0)
    assert shape == (30, 36)
    assert len(grid) == 30
    assert len(grid[0]) == 36
    # Peak value should be normalized to 1.0
    max_v = max(max(row) for row in grid)
    assert abs(max_v - 1.0) < 0.05

    # Bimodal (Ambiguous) belief grid with wall masking
    modes = [
        Point2D(x=3.0, y=2.0, weight=0.55),
        Point2D(x=3.0, y=8.0, weight=0.45),
    ]
    walls = [WallSegment(x1=5.0, y1=0.0, x2=5.0, y2=10.0)]
    b_grid, _ = compute_belief_grid(
        x=3.0, y=2.0, sigma=0.6, plausible_locations=modes, walls=walls
    )
    assert max(max(row) for row in b_grid) > 0.9


def test_pipeline_service_all_12_usps_and_controls():
    svc = GhostTrackPipelineService()
    msg1 = svc.tick()
    assert msg1.state in ("Tracked", "Ambiguous", "Ghost", "Lost")
    assert 0.0 <= msg1.confidence_within_2m <= 1.0
    assert len(msg1.competing_trackers) == 3
    assert len(msg1.scored_readings) == 4

    # Trigger blackout -> should enter Ghost state and populate recovery_advice
    svc.apply_control_command({"command": "trigger_blackout", "blackout": True})
    msg_blackout = svc.tick()
    assert msg_blackout.anchors_online == 0
    assert msg_blackout.state == "Ghost"
    assert len(msg_blackout.recovery_advice) == 4
    assert msg_blackout.recovery_advice[0].priority == 1

    # Restore anchors -> should enter Controlled Recovery check
    svc.apply_control_command({"command": "trigger_blackout", "blackout": False})
    msg_rec = svc.tick()
    assert msg_rec.anchors_online == 4
    assert len(msg_rec.audit_events) > 0
