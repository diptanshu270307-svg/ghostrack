"""
Offline Training & Multi-Tracker Evaluation Script (Stage 3, USPs #03, #09).
1. Generates synthetic sessions across multiple distinct room layouts & wall configurations.
2. Trains Model 1: Self-Trained Reading Scorer & Bias Estimator (model/weights/reading_scorer.pt).
3. Trains Model 2: CNN-LSTM Sequential Tracker with Huber Loss (model/weights/cnn_lstm_tracker.pt).
4. Evaluates on an UNSEEN room layout (with NLOS walls + noise + blackout intervals),
   comparing:
     - Naive Trilateration
     - Geometric Filter + Trilateration
     - Full GhostTrack Pipeline (Geo Engine + AI Scorer + CNN-LSTM + Ghost Mode)
5. Saves real measured metrics (RMSE, p50, p90, empirical 2m reliability) to data/evaluation_report.json.
"""

import os
import json
import math
import random
from typing import List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from shared.schemas import RoomLayout, AnchorConfig, WallSegment
from simulator.sim import SensorSimulator
from engine.contradiction import GeometricContradictionEngine, naive_trilaterate
from model.scorer import ReadingScorer, ReadingScorerNet
from model.tracker import CNNLSTMTracker, CNNLSTMTrackerNet, HuberUncertaintyLoss
from model.ghost import GhostModeController


def make_room_configs() -> List[RoomLayout]:
    """Create 3 distinct training rooms and 1 unseen validation room."""
    room_a = RoomLayout(
        room_id="train_warehouse",
        width=14.0,
        height=10.0,
        anchors=[
            AnchorConfig(id="A1", x=1.0, y=1.0),
            AnchorConfig(id="A2", x=13.0, y=1.0),
            AnchorConfig(id="A3", x=13.0, y=9.0),
            AnchorConfig(id="A4", x=1.0, y=9.0),
        ],
        walls=[
            WallSegment(x1=6.0, y1=0.0, x2=6.0, y2=6.0),
            WallSegment(x1=9.0, y1=4.0, x2=9.0, y2=10.0),
        ],
    )
    room_b = RoomLayout(
        room_id="train_hospital",
        width=10.0,
        height=12.0,
        anchors=[
            AnchorConfig(id="A1", x=0.8, y=0.8),
            AnchorConfig(id="A2", x=9.2, y=0.8),
            AnchorConfig(id="A3", x=9.2, y=11.2),
            AnchorConfig(id="A4", x=0.8, y=11.2),
        ],
        walls=[
            WallSegment(x1=0.0, y1=5.5, x2=6.5, y2=5.5),
            WallSegment(x1=4.5, y1=8.5, x2=10.0, y2=8.5),
        ],
    )
    room_c = RoomLayout(
        room_id="unseen_test_hall",
        width=12.0,
        height=10.0,
        anchors=[
            AnchorConfig(id="A1", x=1.0, y=1.0),
            AnchorConfig(id="A2", x=11.0, y=1.0),
            AnchorConfig(id="A3", x=11.0, y=9.0),
            AnchorConfig(id="A4", x=1.0, y=9.0),
        ],
        walls=[
            WallSegment(x1=5.0, y1=0.0, x2=5.0, y2=5.5),
            WallSegment(x1=5.0, y1=5.5, x2=9.0, y2=5.5),
        ],
    )
    return [room_a, room_b, room_c]


def train_models(steps_per_room: int = 350, epochs: int = 25) -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    rooms = make_room_configs()[:2]  # Train only on first 2 rooms; 3rd is strictly unseen
    scorer_X, scorer_y_clean, scorer_y_bias = [], [], []
    seq_X, seq_Y = [], []

    temp_scorer = ReadingScorer(weights_path="nonexistent.pt")

    for r_idx, layout in enumerate(rooms):
        sim = SensorSimulator(
            layout=layout,
            noise_sigma=0.18,
            nlos_bias=2.6,
            rate_hz=10.0,
            seed=100 + r_idx,
        )
        engine = GeometricContradictionEngine(layout.anchors, layout.width, layout.height)
        window_buf: List[List[float]] = []
        last_xy = (layout.width / 2.0, layout.height / 2.0)

        for s in range(steps_per_room):
            v_ts = 1000.0 + s * 0.1
            (tx, ty), imu = sim.trajectory.step(0.1)
            imu.ts = v_ts
            readings_payload, meta = sim.compute_anchor_readings((tx, ty), v_ts)
            eng_out = engine.process(v_ts, readings_payload.readings, prev_pos=last_xy)

            # Collect ReadingScorer training samples
            for sr in eng_out.scored_readings:
                if not sr.online:
                    continue
                feat = temp_scorer.extract_features(sr)
                m = meta[sr.anchor_id]
                is_clean = 0.0 if m["is_nlos"] else 1.0
                true_bias = max(0.0, sr.distance - m["true_dist"])
                scorer_X.append(feat)
                scorer_y_clean.append(is_clean)
                scorer_y_bias.append(true_bias)

            # Collect CNN-LSTM sequence samples
            obs_x = eng_out.consensus_point.x if eng_out.consensus_point else last_xy[0]
            obs_y = eng_out.consensus_point.y if eng_out.consensus_point else last_xy[1]
            vx = (obs_x - last_xy[0]) / 0.1
            vy = (obs_y - last_xy[1]) / 0.1
            last_xy = (obs_x, obs_y)

            step_feat = [
                obs_x / layout.width,
                obs_y / layout.height,
                vx / 3.0,
                vy / 3.0,
                eng_out.confidence,
                eng_out.anchors_visible / 4.0,
                imu.ax / 5.0,
                imu.ay / 5.0,
                imu.gz / 3.0,
                0.1,
            ]
            window_buf.append(step_feat)
            if len(window_buf) >= 20:
                seq_X.append(list(window_buf[-20:]))
                # Target fine local smoothing residual in normalized room coordinates
                res_dx = float(np.clip((tx - obs_x) / layout.width, -0.025, 0.025))
                res_dy = float(np.clip((ty - obs_y) / layout.height, -0.025, 0.025))
                seq_Y.append([res_dx, res_dy])

    os.makedirs("model/weights", exist_ok=True)

    # 1. Train ReadingScorerNet
    scorer_net = ReadingScorerNet()
    opt_s = optim.Adam(scorer_net.parameters(), lr=3e-3)
    bce = nn.BCELoss()
    huber = nn.HuberLoss(delta=0.5)

    X_s = torch.tensor(scorer_X, dtype=torch.float32)
    y_c = torch.tensor(scorer_y_clean, dtype=torch.float32)
    y_b = torch.tensor(scorer_y_bias, dtype=torch.float32)

    scorer_net.train()
    for _ in range(epochs):
        opt_s.zero_grad()
        pred_c, pred_b = scorer_net(X_s)
        loss_s = bce(pred_c, y_c) + 0.5 * huber(pred_b, y_b)
        loss_s.backward()
        opt_s.step()

    torch.save(scorer_net.state_dict(), "model/weights/reading_scorer.pt")
    print("[Train] Saved ReadingScorer weights -> model/weights/reading_scorer.pt")

    # 2. Train CNNLSTMTrackerNet
    tracker_net = CNNLSTMTrackerNet()
    opt_t = optim.Adam(tracker_net.parameters(), lr=1e-3)
    loss_fn = HuberUncertaintyLoss(delta=0.02)

    X_t = torch.tensor(seq_X, dtype=torch.float32)
    Y_t = torch.tensor(seq_Y, dtype=torch.float32)

    tracker_net.train()
    for _ in range(epochs):
        opt_t.zero_grad()
        pred_delta, log_var = tracker_net(X_t)
        loss_t = loss_fn(pred_delta, Y_t, log_var)
        loss_t.backward()
        opt_t.step()

    torch.save(tracker_net.state_dict(), "model/weights/cnn_lstm_tracker.pt")
    print("[Train] Saved CNN-LSTM Tracker weights -> model/weights/cnn_lstm_tracker.pt")


def evaluate_unseen_room(num_steps: int = 300) -> Dict[str, dict]:
    """
    Evaluate competing trackers on an unseen room layout under:
      - NLOS wall bounces (bias = 2.8m)
      - Gaussian noise (sigma = 0.18m)
      - A 2.5-second anchor blackout window (steps 150..175) to test Ghost Mode!
    """
    unseen_layout = make_room_configs()[2]
    sim = SensorSimulator(
        layout=unseen_layout,
        noise_sigma=0.18,
        nlos_bias=2.8,
        rate_hz=10.0,
        seed=777,
    )
    engine = GeometricContradictionEngine(
        unseen_layout.anchors, unseen_layout.width, unseen_layout.height
    )
    scorer = ReadingScorer(weights_path="model/weights/reading_scorer.pt")
    tracker = CNNLSTMTracker(
        room_width=unseen_layout.width,
        room_height=unseen_layout.height,
        weights_path="model/weights/cnn_lstm_tracker.pt",
        anchor_map=engine.anchor_map,
    )
    ghost = GhostModeController(
        room_width=unseen_layout.width,
        room_height=unseen_layout.height,
        walls=unseen_layout.walls,
    )

    naive_errors: List[float] = []
    geo_errors: List[float] = []
    ghosttrack_errors: List[float] = []
    predicted_conf_2m: List[float] = []
    empirical_within_2m: List[int] = []

    naive_available = 0
    geo_available = 0
    gt_available = 0

    last_naive = (unseen_layout.width / 2.0, unseen_layout.height / 2.0)
    last_geo = (unseen_layout.width / 2.0, unseen_layout.height / 2.0)

    for s in range(num_steps):
        v_ts = 2000.0 + s * 0.1
        # Trigger blackout during steps 150..174 (2.5 seconds of 0 anchors)
        if 150 <= s < 175:
            sim.trigger_blackout(True)
        else:
            sim.trigger_blackout(False)

        (tx, ty), imu = sim.trajectory.step(0.1)
        imu.ts = v_ts
        readings_payload, _ = sim.compute_anchor_readings((tx, ty), v_ts)

        # 1. Naive Trilateration
        n_xy = naive_trilaterate(
            readings_payload.readings,
            engine.anchor_map,
            (unseen_layout.width, unseen_layout.height),
        )
        if n_xy is not None:
            naive_available += 1
            last_naive = n_xy
        naive_errors.append(math.hypot(last_naive[0] - tx, last_naive[1] - ty))

        # 2. Geometric Filter + Trilateration
        eng_out = engine.process(v_ts, readings_payload.readings, prev_pos=last_geo)
        if eng_out.consensus_point is not None:
            geo_available += 1
            last_geo = (eng_out.consensus_point.x, eng_out.consensus_point.y)
        geo_errors.append(math.hypot(last_geo[0] - tx, last_geo[1] - ty))

        # 3. Full GhostTrack Pipeline (Geo + AI Scorer + CNN-LSTM + Ghost Mode)
        eng_out.scored_readings = scorer.score_and_correct(eng_out.scored_readings)
        t_x, t_y, t_sigma, t_vel = tracker.step(eng_out, imu, total_anchors=len(unseen_layout.anchors))
        gx, gy, g_sigma, state, _, conf_2m, _, _ = ghost.step(
            v_ts, eng_out, (t_x, t_y), t_sigma, t_vel, imu
        )
        if state != "Lost":
            gt_available += 1
        err_gt = math.hypot(gx - tx, gy - ty)
        ghosttrack_errors.append(err_gt)
        predicted_conf_2m.append(conf_2m)
        empirical_within_2m.append(1 if err_gt <= 2.0 else 0)

    def summarize(errs: List[float], avail: int) -> dict:
        arr = np.array(errs, dtype=float)
        return {
            "rmse_m": round(float(np.sqrt(np.mean(arr ** 2))), 3),
            "median_p50_m": round(float(np.percentile(arr, 50)), 3),
            "p90_error_m": round(float(np.percentile(arr, 90)), 3),
            "availability_pct": round(100.0 * avail / num_steps, 1),
        }

    report = {
        "scenario": "Unseen 12x10m Room (NLOS bias=2.8m, noise=0.18m, 2.5s Full Blackout)",
        "steps_evaluated": num_steps,
        "naive_trilateration": summarize(naive_errors, naive_available),
        "geometric_filter_trilateration": summarize(geo_errors, geo_available),
        "ghosttrack_full_pipeline": summarize(ghosttrack_errors, gt_available),
        "calibration_usp03": {
            "mean_predicted_prob_within_2m": round(float(np.mean(predicted_conf_2m)), 3),
            "empirical_fraction_within_2m": round(float(np.mean(empirical_within_2m)), 3),
        },
    }

    os.makedirs("data", exist_ok=True)
    with open("data/evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== Unseen Room Evaluation Report (Measured Metrics) ===")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    train_models()
    evaluate_unseen_room()
