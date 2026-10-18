"""
Unit tests for Self-Trained Reading Scorer, CNN-LSTM Tracker, and Ghost Mode Controller.
Verifies monotonic uncertainty growth in Ghost Mode, transition to 'Lost' after prolonged
outage (USP #05), and Controlled Recovery verification before re-trusting fixes (USP #06).
"""

import pytest
from shared.schemas import (
    Point2D,
    AnchorReading,
    ScoredReading,
    EngineOutput,
    IMUPayload,
    WallSegment,
)
from model.scorer import ReadingScorer
from model.tracker import CNNLSTMTracker
from model.ghost import GhostModeController, compute_confidence_within_radius


def test_reading_scorer_bias_and_score():
    scorer = ReadingScorer()
    sr_clean = ScoredReading(
        anchor_id="A1",
        distance=4.0,
        corrected_distance=4.0,
        rssi=-53.0,
        online=True,
        is_liar=False,
        residual=0.05,
        reliability_score=0.95,
    )
    sr_liar = ScoredReading(
        anchor_id="A2",
        distance=9.5,
        corrected_distance=6.5,
        rssi=-82.0,
        online=True,
        is_liar=True,
        residual=3.0,
        reliability_score=0.10,
        estimated_bias=3.0,
    )

    out = scorer.score_and_correct([sr_clean, sr_liar])
    assert out[0].reliability_score > 0.70
    assert out[1].reliability_score < 0.35
    assert out[1].estimated_bias > 1.0
    assert out[1].corrected_distance < out[1].distance


def test_ghost_mode_uncertainty_growth_and_lost_transition():
    """
    USP #05: When 0 anchors are visible:
      1. State transitions to 'Ghost'.
      2. Uncertainty sigma grows monotonically at each step.
      3. After lost_timeout_sec (e.g. 5.0s), state transitions to 'Lost' and stops moving.
    """
    ghost = GhostModeController(
        room_width=12.0,
        room_height=10.0,
        walls=[WallSegment(x1=5.0, y1=0.0, x2=5.0, y2=5.0)],
        lost_timeout_sec=2.0,
        required_recovery_frames=2,
    )

    # First step with 3 visible anchors -> Tracked
    eng_tracked = EngineOutput(
        ts=100.0,
        clean_readings=[AnchorReading(anchor_id="A1", distance=3.0)],
        scored_readings=[],
        rejected=[],
        anchors_visible=3,
        confidence=0.92,
        consensus_point=Point2D(x=3.0, y=3.0),
        plausible_locations=[Point2D(x=3.0, y=3.0)],
        is_ambiguous=False,
    )
    imu = IMUPayload(ts=100.0, ax=0.1, ay=0.0, gz=0.0)
    x, y, sigma_0, state, verified, conf_2m, _, _ = ghost.step(
        100.0, eng_tracked, (3.0, 3.0), 0.30, (1.0, 0.5), imu
    )
    assert state == "Tracked"
    assert verified is True
    assert conf_2m > 0.80

    # Blackout begins at t=100.5 -> Ghost Mode
    eng_blackout = EngineOutput(
        ts=100.5,
        clean_readings=[],
        scored_readings=[],
        rejected=[],
        anchors_visible=0,
        confidence=0.0,
        consensus_point=None,
        plausible_locations=[],
        is_ambiguous=False,
    )
    _, _, sigma_1, state_1, _, _, _, _ = ghost.step(
        100.5, eng_blackout, (3.0, 3.0), 0.30, (1.0, 0.5), imu
    )
    assert state_1 == "Ghost"

    # Advance 1.0s in blackout -> sigma must grow monotonically
    eng_blackout.ts = 101.5
    _, _, sigma_2, state_2, _, _, _, _ = ghost.step(
        101.5, eng_blackout, (3.0, 3.0), 0.30, (1.0, 0.5), imu
    )
    assert state_2 == "Ghost"
    assert sigma_2 > sigma_1

    # Advance past lost_timeout_sec (2.0s -> t=103.0) -> State must become 'Lost'
    eng_blackout.ts = 103.0
    x_lost, y_lost, sigma_lost, state_lost, _, _, _, _ = ghost.step(
        103.0, eng_blackout, (3.0, 3.0), 0.30, (1.0, 0.5), imu
    )
    assert state_lost == "Lost"
    assert sigma_lost > sigma_2

    # USP #06: Controlled Recovery when anchors return at t=103.5
    # Frame 1 of return: should NOT immediately trust as 'Tracked' (recovery_verified=False)
    eng_return_1 = EngineOutput(
        ts=103.5,
        clean_readings=[AnchorReading(anchor_id="A1", distance=4.0)],
        scored_readings=[],
        rejected=[],
        anchors_visible=3,
        confidence=0.90,
        consensus_point=Point2D(x=4.5, y=4.0),
        plausible_locations=[Point2D(x=4.5, y=4.0)],
        is_ambiguous=False,
    )
    _, _, _, state_rec1, verified_1, _, _, _ = ghost.step(
        103.5, eng_return_1, (4.5, 4.0), 0.32, (0.5, 0.2), imu
    )
    assert verified_1 is False
    assert state_rec1 == "Ambiguous"

    # Frame 2 of return (consistent fix near (4.6, 4.1)): now verified -> 'Tracked'!
    eng_return_2 = EngineOutput(
        ts=103.6,
        clean_readings=[AnchorReading(anchor_id="A1", distance=4.0)],
        scored_readings=[],
        rejected=[],
        anchors_visible=3,
        confidence=0.92,
        consensus_point=Point2D(x=4.6, y=4.1),
        plausible_locations=[Point2D(x=4.6, y=4.1)],
        is_ambiguous=False,
    )
    _, _, _, state_rec2, verified_2, _, _, _ = ghost.step(
        103.6, eng_return_2, (4.6, 4.1), 0.32, (0.5, 0.2), imu
    )
    assert verified_2 is True
    assert state_rec2 == "Tracked"
