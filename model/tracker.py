"""
CNN-LSTM Sequential Tracker (PPT Stage 03b).
Integrates the exact 36-feature `CNNLSTM` architecture from `ghosttrack.py`
(`ghosttrack_model.pt`, WINDOW=40, SCALE=25.0, MAX_A=6) alongside the real-time
sliding window momentum filter so live WebSocket streaming and offline benchmarks
share the same core PyTorch architecture.
"""

from typing import Dict, List, Tuple, Optional
from collections import deque
import os
import math
import numpy as np
import torch
import torch.nn as nn

from ghosttrack import build_model, MAX_A, SCALE, WINDOW, multilaterate
from shared.schemas import EngineOutput, IMUPayload


class HuberUncertaintyLoss(nn.Module):
    """
    Heteroscedastic Huber Loss combining outlier-resilient Huber regression
    with learned error-radius uncertainty calibration (matching ghosttrack.py).
    """

    def __init__(self, delta: float = 0.04) -> None:
        super().__init__()
        self.huber = nn.HuberLoss(reduction="none", delta=delta)

    def forward(
        self,
        pred_xy: torch.Tensor,
        target_xy: torch.Tensor,
        log_var: torch.Tensor,
    ) -> torch.Tensor:
        log_var_clamped = torch.clamp(log_var, min=-5.0, max=2.0)
        huber_err = self.huber(pred_xy, target_xy).sum(dim=-1)
        target_log_var = torch.log(torch.clamp(huber_err.detach() * 20.0 + 0.08, min=0.04, max=4.0))
        var_loss = nn.functional.mse_loss(log_var_clamped, target_log_var)
        return huber_err.mean() * 10.0 + 0.2 * var_loss


class CNNLSTMTrackerNet(nn.Module):
    """
    1D Convolution + LSTM hybrid network for sequential indoor trajectory tracking.
    """

    def __init__(
        self,
        in_channels: int = 10,
        conv_channels: int = 32,
        lstm_hidden: int = 64,
        lstm_layers: int = 2,
    ) -> None:
        super().__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels, conv_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.1 if lstm_layers > 1 else 0.0,
        )
        self.pos_head = nn.Sequential(
            nn.Linear(lstm_hidden, 32),
            nn.GELU(),
            nn.Linear(32, 2),
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(lstm_hidden, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        with torch.no_grad():
            nn.init.zeros_(self.pos_head[-1].weight)
            nn.init.zeros_(self.pos_head[-1].bias)
            nn.init.zeros_(self.logvar_head[-1].weight)
            nn.init.constant_(self.logvar_head[-1].bias, -2.2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_conv = self.conv1d(x.transpose(1, 2))
        lstm_out, _ = self.lstm(x_conv.transpose(1, 2))
        last_hidden = lstm_out[:, -1, :]
        delta_xy = torch.tanh(self.pos_head(last_hidden)) * 0.03
        log_var = self.logvar_head(last_hidden).squeeze(-1)
        return delta_xy, log_var


class CNNLSTMTracker:
    """
    Stateful sliding-window CNN-LSTM tracker supporting both the 36-feature
    `ghosttrack_model.pt` (from `ghosttrack.py`) and the 10-feature residual refinement net.
    """

    def __init__(
        self,
        window_size: int = 20,
        room_width: float = 12.0,
        room_height: float = 10.0,
        weights_path: str = "model/weights/cnn_lstm_tracker.pt",
        anchor_map: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        self.window_size = window_size
        self.room_width = room_width
        self.room_height = room_height
        self.weights_path = weights_path
        self.anchor_map = anchor_map or {}

        self.device = torch.device("cpu")
        self.net = CNNLSTMTrackerNet().to(self.device)
        self.trained_loaded = False

        if os.path.exists(weights_path):
            try:
                state = torch.load(weights_path, map_location=self.device, weights_only=True)
                self.net.load_state_dict(state)
                self.trained_loaded = True
            except Exception:
                pass
        self.net.eval()

        # Also load ghosttrack_model.pt (36 features, WINDOW=40) if present
        self.gt_net = build_model(n_feat=36, hidden=96).to(self.device)
        self.gt_loaded = False
        if os.path.exists("ghosttrack_model.pt"):
            try:
                gt_state = torch.load("ghosttrack_model.pt", map_location=self.device, weights_only=True)
                self.gt_net.load_state_dict(gt_state)
                self.gt_loaded = True
                self.trained_loaded = True
            except Exception:
                pass
        self.gt_net.eval()

        self.history: deque = deque(maxlen=window_size)
        self.gt_history: deque = deque(maxlen=WINDOW)
        self.last_pos: Optional[Tuple[float, float]] = None
        self.last_ts: Optional[float] = None
        self.vx: float = 0.0
        self.vy: float = 0.0

    def reset(self) -> None:
        self.history.clear()
        self.gt_history.clear()
        self.last_pos = None
        self.last_ts = None
        self.vx = 0.0
        self.vy = 0.0

    def update_room_size(
        self,
        width: float,
        height: float,
        anchor_map: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        self.room_width = width
        self.room_height = height
        if anchor_map is not None:
            self.anchor_map = anchor_map

    def _build_36_feature_step(
        self,
        engine_out: EngineOutput,
        imu: Optional[IMUPayload],
    ) -> List[float]:
        """Build the 36-element feature vector matching ghosttrack.py::build_windows."""
        feat36: List[float] = []
        anchor_ids = sorted(self.anchor_map.keys())
        scored_by_id = {s.anchor_id: s for s in engine_out.scored_readings}

        for a in range(MAX_A):
            if a < len(anchor_ids):
                aid = anchor_ids[a]
                ax, ay = self.anchor_map[aid]
                sr = scored_by_id.get(aid)
                if sr is not None and sr.online:
                    vis = 1.0
                    d_norm = sr.corrected_distance / SCALE
                    wt = sr.reliability_score
                else:
                    vis = 0.0
                    d_norm = 0.0
                    wt = 0.0
                feat36.extend([d_norm * vis, vis, ax / SCALE, ay / SCALE, wt * vis])
            else:
                feat36.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        if engine_out.consensus_point is not None and engine_out.anchors_visible >= 3:
            geo_x = engine_out.consensus_point.x / SCALE
            geo_y = engine_out.consensus_point.y / SCALE
            has_geo = 1.0
        else:
            geo_x, geo_y, has_geo = 0.0, 0.0, 0.0

        acc_x = (imu.ax if imu else 0.0) / 3.0
        acc_y = (imu.ay if imu else 0.0) / 3.0
        gyro_z = (imu.gz if imu else 0.0) / 2.0
        feat36.extend([geo_x, geo_y, has_geo, acc_x, acc_y, gyro_z])
        return feat36

    def step(
        self,
        engine_out: EngineOutput,
        imu: Optional[IMUPayload],
        total_anchors: int = 4,
    ) -> Tuple[float, float, float, Tuple[float, float]]:
        ts = engine_out.ts
        dt = 0.1
        if self.last_ts is not None and ts > self.last_ts:
            dt = min(1.0, max(0.01, ts - self.last_ts))
        self.last_ts = ts

        online_scored = [
            s for s in engine_out.scored_readings
            if s.online and s.anchor_id in self.anchor_map
        ]
        if len(online_scored) >= 3 and engine_out.consensus_point is not None:
            coords = np.array([self.anchor_map[s.anchor_id] for s in online_scored], dtype=float)
            corr_dists = np.array([s.corrected_distance for s in online_scored], dtype=float)
            weights = np.array([s.reliability_score ** 2 for s in online_scored], dtype=float)
            init_xy = np.array([engine_out.consensus_point.x, engine_out.consensus_point.y])
            refined_xy = multilaterate(coords, corr_dists, w=weights, p0=init_xy)
            obs_x = float(np.clip(refined_xy[0], 0.0, self.room_width))
            obs_y = float(np.clip(refined_xy[1], 0.0, self.room_height))
        elif engine_out.consensus_point is not None:
            obs_x = engine_out.consensus_point.x
            obs_y = engine_out.consensus_point.y
        elif self.last_pos is not None:
            obs_x, obs_y = self.last_pos
        else:
            obs_x, obs_y = self.room_width / 2.0, self.room_height / 2.0

        # Run 36-feature ghosttrack.py CNNLSTM if loaded
        gt_rad_sigma: Optional[float] = None
        if self.gt_loaded and self.anchor_map:
            f36 = self._build_36_feature_step(engine_out, imu)
            self.gt_history.append(f36)
            seq36 = list(self.gt_history)
            while len(seq36) < WINDOW:
                seq36.insert(0, seq36[0])
            x36 = torch.tensor([seq36], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                gt_pos, gt_rad = self.gt_net(x36)
            gt_x = float(gt_pos[0, 0].item()) * SCALE
            gt_y = float(gt_pos[0, 1].item()) * SCALE
            gt_rad_sigma = float(gt_rad[0].item()) * SCALE
            # Blend ghosttrack CNN-LSTM prediction when anchors are sparse
            if engine_out.anchors_visible < 3:
                obs_x = 0.75 * obs_x + 0.25 * float(np.clip(gt_x, 0.2, self.room_width - 0.2))
                obs_y = 0.75 * obs_y + 0.25 * float(np.clip(gt_y, 0.2, self.room_height - 0.2))

        if self.last_pos is not None:
            pred_x = self.last_pos[0] + self.vx * dt
            pred_y = self.last_pos[1] + self.vy * dt
            innovation_dist = math.hypot(obs_x - pred_x, obs_y - pred_y)

            jump_penalty = 1.0 / (1.0 + (max(0.0, innovation_dist - 0.6) / 0.8) ** 2)
            effective_conf = engine_out.confidence * jump_penalty

            gain_pos = max(0.20, min(0.82, effective_conf))
            smooth_x = gain_pos * obs_x + (1.0 - gain_pos) * pred_x
            smooth_y = gain_pos * obs_y + (1.0 - gain_pos) * pred_y

            inst_vx = (smooth_x - self.last_pos[0]) / dt
            inst_vy = (smooth_y - self.last_pos[1]) / dt
            speed = math.hypot(inst_vx, inst_vy)
            if speed > 3.0:
                inst_vx *= 3.0 / speed
                inst_vy *= 3.0 / speed

            alpha_v = max(0.15, min(0.55, 0.5 * effective_conf))
            self.vx = (1.0 - alpha_v) * self.vx + alpha_v * inst_vx
            self.vy = (1.0 - alpha_v) * self.vy + alpha_v * inst_vy
        else:
            smooth_x, smooth_y = obs_x, obs_y

        ax = imu.ax if imu is not None else 0.0
        ay = imu.ay if imu is not None else 0.0
        gz = imu.gz if imu is not None else 0.0

        feat = [
            smooth_x / max(1.0, self.room_width),
            smooth_y / max(1.0, self.room_height),
            self.vx / 3.0,
            self.vy / 3.0,
            engine_out.confidence,
            engine_out.anchors_visible / max(1, total_anchors),
            ax / 5.0,
            ay / 5.0,
            gz / 3.0,
            dt,
        ]
        self.history.append(feat)

        seq = list(self.history)
        while len(seq) < self.window_size:
            seq.insert(0, seq[0])

        x_tensor = torch.tensor([seq], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            delta_xy, log_var = self.net(x_tensor)

        dx = float(delta_xy[0, 0].item()) * self.room_width
        dy = float(delta_xy[0, 1].item()) * self.room_height
        nn_sigma = math.sqrt(math.exp(float(log_var[0].item())))
        if gt_rad_sigma is not None:
            nn_sigma = 0.5 * nn_sigma + 0.5 * min(3.5, max(0.2, gt_rad_sigma))

        final_x = float(np.clip(smooth_x + dx, 0.2, self.room_width - 0.2))
        final_y = float(np.clip(smooth_y + dy, 0.2, self.room_height - 0.2))
        self.last_pos = (final_x, final_y)

        base_sigma = 0.25 + 0.70 * (1.0 - engine_out.confidence)
        if engine_out.is_ambiguous:
            base_sigma = max(base_sigma, 0.90)
        sigma = float(np.clip(0.45 * nn_sigma + 0.55 * base_sigma, 0.20, 4.5))

        return round(final_x, 3), round(final_y, 3), round(sigma, 3), (round(self.vx, 3), round(self.vy, 3))
