"""
Unit tests for the Geometric Contradiction Engine and Explainable Anchor Recovery Adviser.
Includes hand-built geometries with clear NLOS liars, bimodal ambiguity, and staleness.
"""

import math
import pytest
from shared.schemas import AnchorConfig, AnchorReading, RoomLayout, WallSegment
from engine.contradiction import (
    GeometricContradictionEngine,
    naive_trilaterate,
)
from engine.adviser import rank_recovery_anchors


def get_test_anchors():
    return [
        AnchorConfig(id="A1", x=1.0, y=1.0, online=True),
        AnchorConfig(id="A2", x=11.0, y=1.0, online=True),
        AnchorConfig(id="A3", x=11.0, y=9.0, online=True),
        AnchorConfig(id="A4", x=1.0, y=9.0, online=True),
    ]


def test_one_clear_liar_rejection_and_downweighting():
    """
    Hand-built geometry: Target at (5.0, 4.0).
    A1, A3, A4 report true Euclidean distance.
    A2 has a +3.5m NLOS wall-bounce inflation (liar).
    """
    anchors = get_test_anchors()
    engine = GeometricContradictionEngine(anchors, room_width=12.0, room_height=10.0)

    tx, ty = 5.0, 4.0
    d1 = math.hypot(tx - 1.0, ty - 1.0)   # 5.0
    d2_true = math.hypot(tx - 11.0, ty - 1.0)  # 6.708
    d3 = math.hypot(tx - 11.0, ty - 9.0)  # 7.810
    d4 = math.hypot(tx - 1.0, ty - 9.0)   # 6.403

    readings = [
        AnchorReading(anchor_id="A1", distance=d1, rssi=-55.0, online=True, ts=100.0),
        AnchorReading(anchor_id="A2", distance=d2_true + 3.5, rssi=-78.0, online=True, ts=100.0),
        AnchorReading(anchor_id="A3", distance=d3, rssi=-62.0, online=True, ts=100.0),
        AnchorReading(anchor_id="A4", distance=d4, rssi=-58.0, online=True, ts=100.0),
    ]

    out = engine.process(ts=100.0, readings=readings)

    # A2 must be identified as the liar and downweighted, NOT silently erased from scored_readings
    rejected_ids = [r.anchor_id for r in out.rejected]
    assert "A2" in rejected_ids
    assert out.anchors_visible == 3

    a2_scored = next(s for s in out.scored_readings if s.anchor_id == "A2")
    assert a2_scored.is_liar is True
    assert a2_scored.reliability_score < 0.20
    assert a2_scored.residual == pytest.approx(3.5, abs=0.25)
    assert "NLOS wall-bounce" in a2_scored.reason

    # Geometric engine consensus must stay locked near (5.0, 4.0)
    assert out.consensus_point is not None
    err_engine = math.hypot(out.consensus_point.x - tx, out.consensus_point.y - ty)
    assert err_engine < 0.15

    # Naive trilateration is badly corrupted by the liar
    naive_xy = naive_trilaterate(readings, engine.anchor_map)
    assert naive_xy is not None
    err_naive = math.hypot(naive_xy[0] - tx, naive_xy[1] - ty)
    assert err_naive > 1.0


def test_two_anchor_bimodal_ambiguity():
    """
    When only 2 anchors (A1 at (1,1) and A2 at (11,1)) are online,
    or symmetric across y=5: A1 at (2, 5) and A2 at (10, 5) with radius 5.0
    intersect at (6, 8) and (6, 2) inside the room -> Ambiguous!
    """
    anchors = [
        AnchorConfig(id="A1", x=2.0, y=5.0, online=True),
        AnchorConfig(id="A2", x=10.0, y=5.0, online=True),
        AnchorConfig(id="A3", x=6.0, y=9.0, online=False),
        AnchorConfig(id="A4", x=6.0, y=1.0, online=False),
    ]
    engine = GeometricContradictionEngine(anchors, room_width=12.0, room_height=10.0)

    readings = [
        AnchorReading(anchor_id="A1", distance=5.0, online=True, ts=200.0),
        AnchorReading(anchor_id="A2", distance=5.0, online=True, ts=200.0),
        AnchorReading(anchor_id="A3", distance=5.0, online=False, ts=200.0),
        AnchorReading(anchor_id="A4", distance=5.0, online=False, ts=200.0),
    ]

    out = engine.process(ts=200.0, readings=readings)
    assert out.is_ambiguous is True
    assert len(out.plausible_locations) == 2

    ys = sorted([p.y for p in out.plausible_locations])
    assert ys[0] == pytest.approx(2.0, abs=0.1)
    assert ys[1] == pytest.approx(8.0, abs=0.1)

    # Test Explainable Anchor Recovery Adviser on this ambiguous state
    layout = RoomLayout(room_id="test", width=12.0, height=10.0, anchors=anchors, walls=[])
    advice = rank_recovery_anchors(layout, out, (6.0, 8.0))
    assert len(advice) == 2
    assert advice[0].priority == 1
    assert "breaks bimodal ambiguity" in advice[0].reason


def test_zero_anchors_triggers_zero_confidence():
    anchors = get_test_anchors()
    engine = GeometricContradictionEngine(anchors)
    readings = [
        AnchorReading(anchor_id=a.id, distance=5.0, online=False, ts=300.0)
        for a in anchors
    ]
    out = engine.process(ts=300.0, readings=readings)
    assert out.anchors_visible == 0
    assert out.confidence == 0.0
    assert out.consensus_point is None


def test_freshness_aware_staleness_penalty():
    anchors = get_test_anchors()
    engine = GeometricContradictionEngine(anchors, stale_threshold_ms=500.0)
    tx, ty = 6.0, 5.0
    readings = [
        AnchorReading(
            anchor_id=a.id,
            distance=math.hypot(tx - a.x, ty - a.y),
            online=True,
            ts=98.5,  # 1.5 seconds old (1500 ms > 500 ms threshold)
        )
        for a in anchors
    ]
    out = engine.process(ts=100.0, readings=readings)
    for s in out.scored_readings:
        assert s.age_ms == pytest.approx(1500.0, abs=1.0)
        assert s.reliability_score < 0.50
        assert "stale_packet" in s.reason
