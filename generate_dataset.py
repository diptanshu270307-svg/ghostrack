"""
Generates synthetic multi-room indoor positioning datasets for GhostTrack:
  - dataset/train.csv  (unseen room layouts vs val/test)
  - dataset/val.csv
  - dataset/test.csv   (strictly unseen room layouts & anchor geometries)

Columns match ghosttrack.py exactly:
  session_id, t, x, y, n_anchors, acc_x, acc_y, gyro_z,
  d0..d5, a0_x..a5_x, a0_y..a5_y, vis0..vis5, nlos0..nlos5
"""

import os
import math
import numpy as np
import pandas as pd

from shared.schemas import WallSegment
from simulator.geometry import check_line_of_sight, resolve_wall_collision

MAX_A = 6
DT = 0.1  # 10 Hz


def simulate_session(
    session_id: int,
    rng: np.random.Generator,
    steps: int = 240,
) -> pd.DataFrame:
    # Random room dimensions between 10m and 22m (<= SCALE=25m)
    room_w = float(rng.uniform(10.0, 22.0))
    room_h = float(rng.uniform(8.0, 18.0))
    n_anchors = int(rng.integers(4, MAX_A + 1))  # 4, 5, or 6 anchors

    # Place anchors near perimeter / corners with jitter
    perim_pts = [
        (0.8, 0.8),
        (room_w - 0.8, 0.8),
        (room_w - 0.8, room_h - 0.8),
        (0.8, room_h - 0.8),
        (room_w * 0.5, 0.8),
        (room_w * 0.5, room_h - 0.8),
    ]
    anchors = np.full((MAX_A, 2), np.nan, dtype=np.float32)
    for a in range(n_anchors):
        bx, by = perim_pts[a]
        anchors[a, 0] = np.clip(bx + rng.uniform(-0.4, 0.4), 0.5, room_w - 0.5)
        anchors[a, 1] = np.clip(by + rng.uniform(-0.4, 0.4), 0.5, room_h - 0.5)

    # 1 or 2 interior wall segments causing physical NLOS bounces
    walls = [
        WallSegment(
            x1=float(room_w * rng.uniform(0.35, 0.55)),
            y1=0.0,
            x2=float(room_w * rng.uniform(0.35, 0.55)),
            y2=float(room_h * 0.55),
        ),
        WallSegment(
            x1=float(room_w * 0.45),
            y1=float(room_h * 0.55),
            x2=float(room_w * 0.80),
            y2=float(room_h * 0.55),
        ),
    ]

    # Target trajectory initialization
    x = float(rng.uniform(1.5, room_w - 1.5))
    y = float(rng.uniform(1.5, room_h - 1.5))
    theta = float(rng.uniform(0, 2 * math.pi))
    speed = float(rng.uniform(1.0, 1.6))
    omega = 0.0
    vx = speed * math.cos(theta)
    vy = speed * math.sin(theta)

    # Schedule a blind-spot / dropout window in each session (15-25 steps where <3 anchors visible)
    blackout_start = int(rng.integers(60, steps - 40))
    blackout_end = blackout_start + int(rng.integers(15, 25))
    keep_during_blackout = int(rng.integers(0, 3))  # 0, 1, or 2 visible anchors

    rows = []
    for step in range(steps):
        t = round(step * DT, 2)
        prev_vx, prev_vy = vx, vy

        omega = float(np.clip(0.85 * omega + 0.15 * rng.normal(0.0, 0.6), -1.2, 1.2))
        theta = (theta + omega * DT) % (2 * math.pi)
        vx = speed * math.cos(theta)
        vy = speed * math.sin(theta)

        cand_x = x + vx * DT
        cand_y = y + vy * DT
        (x, y), (new_vx, new_vy) = resolve_wall_collision(
            (cand_x, cand_y), (vx, vy), walls, room_w, room_h, margin=0.4
        )
        if new_vx != vx or new_vy != vy:
            vx, vy = new_vx, new_vy
            theta = math.atan2(vy, vx)
            omega = -omega

        ax_imu = (vx - prev_vx) / DT + float(rng.normal(0.0, 0.06))
        ay_imu = (vy - prev_vy) / DT + float(rng.normal(0.0, 0.06))
        gz_imu = omega + float(rng.normal(0.0, 0.015))

        in_blackout = blackout_start <= step < blackout_end

        row = {
            "session_id": session_id,
            "t": t,
            "x": round(x, 4),
            "y": round(y, 4),
            "n_anchors": n_anchors,
            "acc_x": round(ax_imu, 4),
            "acc_y": round(ay_imu, 4),
            "gyro_z": round(gz_imu, 4),
        }

        # At most 1 anchor per step gets a strong NLOS wall bounce so >=3 clean anchors exist in normal coverage
        nlos_candidates = []
        for a in range(n_anchors):
            has_los, _ = check_line_of_sight((x, y), (float(anchors[a, 0]), float(anchors[a, 1])), walls)
            if not has_los:
                nlos_candidates.append(a)
        active_nlos_anchor = int(rng.choice(nlos_candidates)) if (len(nlos_candidates) > 0 and rng.random() < 0.45) else -1

        for a in range(MAX_A):
            if a >= n_anchors:
                row[f"d{a}"] = np.nan
                row[f"a{a}_x"] = np.nan
                row[f"a{a}_y"] = np.nan
                row[f"vis{a}"] = 0
                row[f"nlos{a}"] = 0
                continue

            if in_blackout and a >= keep_during_blackout:
                row[f"d{a}"] = np.nan
                row[f"a{a}_x"] = round(float(anchors[a, 0]), 3)
                row[f"a{a}_y"] = round(float(anchors[a, 1]), 3)
                row[f"vis{a}"] = 0
                row[f"nlos{a}"] = 0
                continue

            true_d = math.hypot(x - float(anchors[a, 0]), y - float(anchors[a, 1]))
            is_nlos = 1 if a == active_nlos_anchor else 0
            bias = float(rng.uniform(1.8, 3.8)) if is_nlos else 0.0
            obs_d = max(0.05, true_d + bias + float(rng.normal(0.0, 0.12)))

            row[f"d{a}"] = round(obs_d, 4)
            row[f"a{a}_x"] = round(float(anchors[a, 0]), 3)
            row[f"a{a}_y"] = round(float(anchors[a, 1]), 3)
            row[f"vis{a}"] = 1
            row[f"nlos{a}"] = is_nlos

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs("dataset", exist_ok=True)
    splits = [
        ("train", 18, 1000),
        ("val", 5, 2000),
        ("test", 6, 3000),
    ]
    for name, n_sessions, base_seed in splits:
        rng = np.random.default_rng(base_seed)
        dfs = [
            simulate_session(session_id=base_seed + i, rng=rng, steps=220)
            for i in range(n_sessions)
        ]
        full_df = pd.concat(dfs, ignore_index=True)
        out_path = f"dataset/{name}.csv"
        full_df.to_csv(out_path, index=False)
        print(f"Created {out_path} ({len(full_df)} rows across {n_sessions} sessions)")


if __name__ == "__main__":
    main()
