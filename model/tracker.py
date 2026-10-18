"""
CNN-LSTM Sequential Tracker (PPT Stage 03b).
Architecture:
  1. Temporal 1D Convolution (nn.Conv1d) -> smooths local high-frequency multipath noise.
  2. Recurrent LSTM (nn.LSTM) -> learns target velocity momentum and trajectory dynamics.
  3. Dual Head -> outputs 2D position refinement (x, y) and log-variance (uncertainty sigma).

Loss Function Rationale (Why Huber Loss):
  - Standard MSE (L2) squares residual NLOS spikes, causing gradient explosion and jumping estimates.
  - Pure MAE (L1) has discontinuous derivatives at zero, causing jitter when locked on target.
  - Huber Loss (smooth L1) behaves quadratically for small errors (|e| < delta) for sub-decimeter
    precision, and linearly for large errors (|e| >= delta) so unrejected multipath outliers
    cannot pull the LSTM momentum off course.
"""

from typing import Dict, List, Tuple, Optional
from collections import deque
import os
import math
import numpy as np
import torch
import torch.nn as nn

from shared.schemas import EngineOutput, IMUPayload
from engine.contradiction import weighted_trilaterate


class HuberUncertaintyLoss(nn.Module):
    """
    Heteroscedastic Huber Loss combining outlier-resilient Huber regression
    with learned log-variance uncertainty calibration.
    """

    def __init__(self, delta: float = 0.03) -> None:
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
        # Direct Huber loss + calibrated variance matching term
        target_log_var = torch.log(torch.clamp(huber_err.detach() * 20.0 + 0.08, min=0.04, max=4.0))
        var_loss = nn.functional.mse_loss(log_var_clamped, target_log_var)
        return huber_err.mean() * 10.0 + 0.2 * var_loss


class CNNLSTMTrackerNet(nn.Module):
    """
    1D Convolution + LSTM hybrid network for sequential indoor trajectory tracking.
    Input shape: (batch, seq_len, in_channels=10)
      Features per timestep:
        [norm_x_obs, norm_y_obs, vx_obs, vy_obs, confidence, anchors_ratio, ax, ay, gz, dt]
    Outputs:
      - delta_xy: residual trajectory smoothing correction in normalized coordinates
      - log_var: log(sigma^2) spatial uncertainty
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
        delta_xy = torch.tanh(self.pos_head(last_hidden)) * 0.03  # Fine local multipath correction
        log_var = self.logvar_head(last_hidden).squeeze(-1)
        return delta_xy, log_var


class CNNLSTMTracker:
    """
    Stateful sliding-window CNN-LSTM tracker that maintains a 20-step history buffer,
    fuses AI-bias-corrected distances, estimates target momentum (vx, vy),
    and outputs smoothed (x, y) + calibrated uncertainty sigma.
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

        self.history: deque = deque(maxlen=window_size)
        self.last_pos: Optional[Tuple[float, float]] = None
        self.last_ts: Optional[float] = None
        self.vx: float = 0.0
        self.vy: float = 0.0

    def reset(self) -> None:
        self.history.clear()
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

    def step(
        self,
        engine_out: EngineOutput,
        imu: Optional[IMUPayload],
        total_anchors: int = 4,
    ) -> Tuple[float, float, float, Tuple[float, float]]:
        """
        Process one frame through the temporal Conv1D + LSTM tracker.
        Returns: (est_x, est_y, sigma, (vx, vy))
        """
        ts = engine_out.ts
        dt = 0.1
        if self.last_ts is not None and ts > self.last_ts:
            dt = min(1.0, max(0.01, ts - self.last_ts))
        self.last_ts = ts

        # If anchor_map and AI-scored readings are available, compute AI-bias-corrected trilateration
        online_scored = [
            s for s in engine_out.scored_readings
            if s.online and s.anchor_id in self.anchor_map
        ]
        if len(online_scored) >= 3 and engine_out.consensus_point is not None:
            coords = np.array([self.anchor_map[s.anchor_id] for s in online_scored], dtype=float)
            corr_dists = np.array([s.corrected_distance for s in online_scored], dtype=float)
            weights = np.array([s.reliability_score ** 2 for s in online_scored], dtype=float)
            init_xy = np.array([engine_out.consensus_point.x, engine_out.consensus_point.y])
            refined_xy, _ = weighted_trilaterate(
                coords, corr_dists, weights, (self.room_width, self.room_height), init_guess=init_xy
            )
            obs_x, obs_y = float(refined_xy[0]), float(refined_xy[1])
        elif engine_out.consensus_point is not None:
            obs_x = engine_out.consensus_point.x
            obs_y = engine_out.consensus_point.y
        elif self.last_pos is not None:
            obs_x, obs_y = self.last_pos
        else:
            obs_x, obs_y = self.room_width / 2.0, self.room_height / 2.0

        # Kinematic momentum filter (Kalman-like alpha-beta update)
        if self.last_pos is not None:
            pred_x = self.last_pos[0] + self.vx * dt
            pred_y = self.last_pos[1] + self.vy * dt
            innovation_dist = math.hypot(obs_x - pred_x, obs_y - pred_y)

            # Downweight sudden multi-meter jumps caused by simultaneous multi-wall NLOS
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

        final_x = float(np.clip(smooth_x + dx, 0.2, self.room_width - 0.2))
        final_y = float(np.clip(smooth_y + dy, 0.2, self.room_height - 0.2))
        self.last_pos = (final_x, final_y)

        base_sigma = 0.25 + 0.70 * (1.0 - engine_out.confidence)
        if engine_out.is_ambiguous:
            base_sigma = max(base_sigma, 0.90)
        sigma = float(np.clip(0.45 * nn_sigma + 0.55 * base_sigma, 0.20, 4.5))

        return round(final_x, 3), round(final_y, 3), round(sigma, 3), (round(self.vx, 3), round(self.vy, 3))
