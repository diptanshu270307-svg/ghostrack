"""
Unit tests for the Simulator, geometry utilities, and IMU synthesis.
"""

import os
import csv
import pytest
from simulator.geometry import (
    segments_intersect,
    point_segment_distance,
    check_line_of_sight,
    resolve_wall_collision,
)
from simulator.trajectory import TrajectoryGenerator
from simulator.sim import SensorSimulator, get_default_layout
from shared.schemas import WallSegment, AnchorConfig, RoomLayout


def test_segment_intersection():
    # Crossing segments
    assert segments_intersect((0, 0), (2, 2), (0, 2), (2, 0)) is True

    # Parallel segments
    assert segments_intersect((0, 0), (0, 2), (1, 0), (1, 2)) is False

    # Disjoint non-parallel segments
    assert segments_intersect((0, 0), (1, 1), (2, 2), (3, 3)) is False

    # T-intersection (touching)
    assert segments_intersect((0, 0), (4, 0), (2, 0), (2, 2)) is True


def test_point_segment_distance():
    p = (1.0, 1.0)
    a = (0.0, 0.0)
    b = (2.0, 0.0)
    dist = point_segment_distance(p, a, b)
    assert pytest.approx(dist, 1e-4) == 1.0

    # Closer to endpoint a
    p2 = (-1.0, 0.0)
    assert pytest.approx(point_segment_distance(p2, a, b), 1e-4) == 1.0


def test_check_line_of_sight():
    walls = [WallSegment(x1=5.0, y1=0.0, x2=5.0, y2=8.0)]
    
    # Target and anchor on opposite sides of wall
    target = (2.0, 4.0)
    anchor = (8.0, 4.0)
    has_los, cuts = check_line_of_sight(target, anchor, walls)
    assert has_los is False
    assert cuts == 1

    # Target and anchor on same side
    anchor_same_side = (3.0, 7.0)
    has_los, cuts = check_line_of_sight(target, anchor_same_side, walls)
    assert has_los is True
    assert cuts == 0


def test_trajectory_generator_bounds_and_imu():
    gen = TrajectoryGenerator(room_width=10.0, room_height=8.0, seed=123)
    
    for _ in range(100):
        (x, y), imu = gen.step(dt=0.1)
        assert 0.0 <= x <= 10.0
        assert 0.0 <= y <= 8.0
        assert isinstance(imu.ax, float)
        assert isinstance(imu.ay, float)
        assert isinstance(imu.gz, float)


def test_simulator_nlos_and_noise():
    layout = RoomLayout(
        room_id="test_room",
        width=10.0,
        height=10.0,
        anchors=[
            AnchorConfig(id="A1", x=1.0, y=5.0, online=True),
            AnchorConfig(id="A2", x=9.0, y=5.0, online=True),
        ],
        walls=[WallSegment(x1=5.0, y1=0.0, x2=5.0, y2=10.0)],
    )

    # Set 0 noise and 3.0m NLOS bias to strictly test geometry
    sim = SensorSimulator(
        layout=layout,
        noise_sigma=0.0,
        nlos_bias=3.0,
        rate_hz=10.0,
        seed=42,
    )

    # Target at (3, 5): A1 is LOS (distance 2.0), A2 is NLOS across wall (true distance 6.0 + 3.0 = 9.0)
    payload, meta = sim.compute_anchor_readings((3.0, 5.0), 100.0)

    r_a1 = next(r for r in payload.readings if r.anchor_id == "A1")
    r_a2 = next(r for r in payload.readings if r.anchor_id == "A2")

    assert pytest.approx(r_a1.distance, 1e-2) == 2.0
    assert meta["A1"]["is_nlos"] is False

    assert pytest.approx(r_a2.distance, 1e-2) == 9.0  # 6.0 true + 3.0 NLOS bias
    assert meta["A2"]["is_nlos"] is True


def test_simulator_toggle_anchor():
    sim = SensorSimulator(seed=42)
    assert sim.anchor_status["A1"] is True

    # Toggle offline
    sim.toggle_anchor("A1", False)
    payload, _ = sim.compute_anchor_readings((2.0, 2.0), 100.0)
    r_a1 = next(r for r in payload.readings if r.anchor_id == "A1")
    assert r_a1.online is False
    assert r_a1.rssi is None

    # Toggle back online
    sim.toggle_anchor("A1", True)
    payload, _ = sim.compute_anchor_readings((2.0, 2.0), 100.0)
    r_a1 = next(r for r in payload.readings if r.anchor_id == "A1")
    assert r_a1.online is True


def test_offline_dataset_generation(tmp_path):
    output_csv = str(tmp_path / "test_offline.csv")
    sim = SensorSimulator(seed=99)
    sim.generate_offline_dataset(num_samples=25, output_csv_path=output_csv, seed=99)

    assert os.path.exists(output_csv)
    with open(output_csv, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        # 25 timesteps * 4 anchors = 100 rows
        assert len(reader) == 100
        assert "observed_distance" in reader[0]
        assert "true_x" in reader[0]
        assert "ax" in reader[0]
