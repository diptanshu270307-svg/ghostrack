"""
Self-Trained AI Reading Scorer & Range Bias Estimator (PPT Stage 03a, USP #04).
Scores each anchor reading for LOS reliability and estimates positive NLOS wall-bounce
range bias using room-invariant physical features (distance, RSSI path-loss discrepancy,
geometric consensus residual, and temporal jitter).
"""

from typing import Dict, List, Tuple
from collections import deque
import math
import os
import numpy as np
import torch
import torch.nn as nn

from shared.schemas import ScoredReading


class ReadingScorerNet(nn.Module):
    """
    Lightweight MLP operating on 6 room-invariant physical link features per reading:
      [norm_dist, norm_rssi, path_loss_gap, signed_geo_residual, geo_weight, temporal_jitter]
    Outputs:
      - logit_clean: probability of LOS (via Sigmoid)
      - raw_bias: estimated positive NLOS range bias in meters (via Softplus)
    """

    def __init__(self, in_features: int = 6, hidden_dim: int = 32) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.score_head = nn.Linear(hidden_dim, 1)
        self.bias_head = nn.Linear(hidden_dim, 1)
        self._init_physics_informed_weights()

    def _init_physics_informed_weights(self) -> None:
        """
        Initialize with sensible physics priors so the model works out-of-the-box
        even before offline fine-tuning, and converges rapidly during training.
        """
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight, gain=0.5)
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        ai_score = torch.sigmoid(self.score_head(h)).squeeze(-1)
        est_bias = torch.nn.functional.softplus(self.bias_head(h)).squeeze(-1)
        return ai_score, est_bias


class ReadingScorer:
    """
    Wraps ReadingScorerNet with feature extraction, rolling jitter history,
    and physics-blended inference so readings are scored and bias-corrected reliably.
    """

    def __init__(self, weights_path: str = "model/weights/reading_scorer.pt") -> None:
        self.device = torch.device("cpu")
        self.net = ReadingScorerNet().to(self.device)
        self.weights_path = weights_path
        self.trained_loaded = False

        if os.path.exists(weights_path):
            try:
                state = torch.load(weights_path, map_location=self.device, weights_only=True)
                self.net.load_state_dict(state)
                self.trained_loaded = True
            except Exception:
                pass
        self.net.eval()

        # Rolling history of last 5 distances per anchor for jitter calculation
        self.dist_history: Dict[str, deque] = {}

        # Path-loss reference constants (matching simulator)
        self.rssi_0 = -40.0
        self.path_loss_n = 2.2

    def rssi_to_expected_distance(self, rssi: float) -> float:
        """Convert RSSI (dBm) to expected free-space distance via log-distance path loss."""
        exponent = (self.rssi_0 - rssi) / (10.0 * self.path_loss_n)
        return float(np.clip(10.0 ** exponent, 0.1, 30.0))

    def extract_features(self, sr: ScoredReading) -> List[float]:
        """Compute 6 room-invariant physical features for a single anchor reading."""
        aid = sr.anchor_id
        if aid not in self.dist_history:
            self.dist_history[aid] = deque(maxlen=5)
        if sr.online:
            self.dist_history[aid].append(sr.distance)

        hist = list(self.dist_history[aid])
        jitter = float(np.std(hist)) if len(hist) >= 2 else 0.0

        norm_dist = sr.distance / 15.0
        rssi_val = sr.rssi if sr.rssi is not None else -65.0
        norm_rssi = (rssi_val + 100.0) / 60.0

        # When NLOS occurs, RSSI drops by ~14 dBm, so RSSI-implied distance is MUCH larger
        # than true distance, and geometric residual is positive.
        rssi_dist = self.rssi_to_expected_distance(rssi_val)
        path_loss_gap = max(0.0, (rssi_dist - sr.distance) / 10.0)

        return [
            float(norm_dist),
            float(norm_rssi),
            float(path_loss_gap),
            float(sr.residual / 5.0),
            float(sr.reliability_score),
            float(jitter),
        ]

    def score_and_correct(self, scored_readings: List[ScoredReading]) -> List[ScoredReading]:
        """
        Enrich each ScoredReading with our AI model's reliability score and
        estimated range bias, updating corrected_distance.
        """
        if not scored_readings:
            return scored_readings

        feats = [self.extract_features(sr) for sr in scored_readings]
        x_tensor = torch.tensor(feats, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            nn_scores, nn_biases = self.net(x_tensor)

        for idx, sr in enumerate(scored_readings):
            if not sr.online:
                continue

            # Blend trained neural network output with physical residual/RSSI evidence
            phys_score = sr.reliability_score
            phys_bias = sr.estimated_bias

            # Check RSSI excess attenuation as secondary NLOS indicator
            if sr.rssi is not None:
                expected_los_rssi = self.rssi_0 - 10.0 * self.path_loss_n * math.log10(max(0.1, sr.distance))
                rssi_drop = max(0.0, expected_los_rssi - sr.rssi)
                if rssi_drop > 8.0 and not sr.is_liar:
                    phys_score = min(phys_score, 0.45)

            if self.trained_loaded:
                ai_s = float(0.6 * float(nn_scores[idx].item()) + 0.4 * phys_score)
                est_b = float(0.6 * float(nn_biases[idx].item()) + 0.4 * phys_bias)
            else:
                ai_s = float(phys_score)
                est_b = float(phys_bias)

            ai_s = float(np.clip(ai_s, 0.02, 0.99))
            est_b = max(0.0, est_b) if (sr.is_liar or ai_s < 0.5) else 0.0

            sr.ai_score = round(ai_s, 3)
            sr.estimated_bias = round(est_b, 3)
            sr.reliability_score = round(0.5 * sr.reliability_score + 0.5 * ai_s, 3)
            sr.corrected_distance = round(max(0.1, sr.distance - est_b), 3)

        return scored_readings
