"""
Ghost-Track Dead-Reckoning & 4-State Lifecycle Controller (USPs #01, #03, #05, #06, #07, #11).
Manages transitions across the 4 PPT states:
  - Tracked   (>= 3 clean anchors, verified fix)
  - Ambiguous (2 clean anchors with bimodal symmetry, or unverified recovery)
  - Ghost     (0-1 anchors: short-term prediction via LSTM momentum + IMU dead-reckoning)
  - Lost      (outage exceeds timeout threshold: halts fictional path extrapolation)
Also enforces Controlled Recovery verification when exiting a dead zone.
"""

from typing import List, Tuple, Optional, Set
from collections import deque
import math

from shared.schemas import (
    TrackingState,
    Point2D,
    WallSegment,
    EngineOutput,
    IMUPayload,
    AuditLogEntry,
)
from simulator.geometry import resolve_wall_collision, check_line_of_sight


def compute_confidence_within_radius(
    sigma: float,
    radius_m: float = 2.0,
    mode_weight: float = 1.0,
) -> float:
    """
    Compute calibrated probability that true target is within `radius_m` (default 2.0m)
    under a 2D isotropic Gaussian belief distribution (Rayleigh radial CDF):
      P(r <= R) = (1 - exp(-R^2 / (2 * sigma^2))) * mode_weight
    """
    s = max(0.15, sigma)
    prob = (1.0 - math.exp(-(radius_m * radius_m) / (2.0 * s * s))) * mode_weight
    return round(float(max(0.05, min(0.99, prob))), 3)


class GhostModeController:
    """
    Fuses LSTM momentum with IMU dead-reckoning during coverage dropouts,
    grows uncertainty monotonically, transitions to 'Lost' after prolonged blackout,
    and verifies returning fixes via Controlled Recovery.
    """

    def __init__(
        self,
        room_width: float = 12.0,
        room_height: float = 10.0,
        walls: Optional[List[WallSegment]] = None,
        lost_timeout_sec: float = 5.0,
        required_recovery_frames: int = 2,
    ) -> None:
        self.room_width = room_width
        self.room_height = room_height
        self.walls = walls or []
        self.lost_timeout_sec = lost_timeout_sec
        self.required_recovery_frames = required_recovery_frames

        self.state: TrackingState = "Tracked"
        self.x: float = room_width * 0.25
        self.y: float = room_height * 0.25
        self.vx: float = 0.0
        self.vy: float = 0.0
        self.sigma: float = 0.35

        self.ghost_start_ts: Optional[float] = None
        self.last_ts: Optional[float] = None
        self.last_valid_measurement_ts: Optional[float] = None

        # Controlled recovery state (USP #06)
        self.recovery_verified: bool = True
        self.recovery_streak: int = 0
        self.last_candidate_fix: Optional[Tuple[float, float]] = None

        # Audit log buffer (USP #11)
        self.audit_log: deque = deque(maxlen=25)
        self._prev_liars: Set[str] = set()
        self._prev_top_advice: Optional[str] = None

    def reset(self) -> None:
        self.state = "Tracked"
        self.x = self.room_width * 0.25
        self.y = self.room_height * 0.25
        self.vx = 0.0
        self.vy = 0.0
        self.sigma = 0.35
        self.ghost_start_ts = None
        self.last_ts = None
        self.last_valid_measurement_ts = None
        self.recovery_verified = True
        self.recovery_streak = 0
        self.last_candidate_fix = None
        self.audit_log.clear()
        self._prev_liars.clear()
        self._prev_top_advice = None

    def update_layout(self, width: float, height: float, walls: List[WallSegment]) -> None:
        self.room_width = width
        self.room_height = height
        self.walls = walls

    def _log_audit(
        self,
        ts: float,
        category: str,
        severity: str,
        message: str,
    ) -> None:
        self.audit_log.appendleft(
            AuditLogEntry(
                ts=round(ts, 3),
                category=category,  # type: ignore[arg-type]
                severity=severity,  # type: ignore[arg-type]
                message=message,
            )
        )

    def step(
        self,
        ts: float,
        engine_out: EngineOutput,
        tracker_xy: Tuple[float, float],
        tracker_sigma: float,
        tracker_vel: Tuple[float, float],
        imu: Optional[IMUPayload],
        top_advice_anchor: Optional[str] = None,
    ) -> Tuple[float, float, float, TrackingState, bool, float, float, List[AuditLogEntry]]:
        """
        Execute one step of the 4-state positioning lifecycle.
        Returns:
          (x, y, sigma, state, recovery_verified, confidence_within_2m, measurement_age_ms, audit_events)
        """
        dt = 0.1
        if self.last_ts is not None and ts > self.last_ts:
            dt = min(1.0, max(0.01, ts - self.last_ts))
        self.last_ts = ts

        prev_state = self.state

        # Log newly detected NLOS liars (USP #04, #11)
        current_liars = {r.anchor_id for r in engine_out.rejected}
        for rej in engine_out.rejected:
            if rej.anchor_id not in self._prev_liars:
                self._log_audit(
                    ts,
                    "LIAR_DOWNWEIGHT",
                    "warn",
                    f"Anchor {rej.anchor_id} downweighted (w={rej.reliability_score:.2f}): {rej.reason}",
                )
        self._prev_liars = current_liars

        # Log recovery adviser priority changes (USP #02, #11)
        if top_advice_anchor and top_advice_anchor != self._prev_top_advice:
            self._log_audit(
                ts,
                "ADVISER",
                "info",
                f"Recovery Adviser: prioritized Anchor {top_advice_anchor} for restoration",
            )
            self._prev_top_advice = top_advice_anchor
        elif not top_advice_anchor:
            self._prev_top_advice = None

        # Check if we have usable anchor coverage (>= 2 visible anchors)
        has_coverage = engine_out.anchors_visible >= 2 and engine_out.consensus_point is not None

        if has_coverage:
            self.last_valid_measurement_ts = ts
            cand_x, cand_y = tracker_xy

            # Controlled Recovery check when returning from Ghost or Lost (USP #06)
            if prev_state in ("Ghost", "Lost") or not self.recovery_verified:
                if self.last_candidate_fix is not None:
                    jump = math.hypot(
                        cand_x - self.last_candidate_fix[0],
                        cand_y - self.last_candidate_fix[1],
                    )
                    if jump <= 1.8:
                        self.recovery_streak += 1
                    else:
                        self.recovery_streak = 1
                else:
                    self.recovery_streak = 1

                self.last_candidate_fix = (cand_x, cand_y)

                if self.recovery_streak < self.required_recovery_frames:
                    self.recovery_verified = False
                    self.state = "Ambiguous"
                    # Smoothly pull estimate halfway toward candidate while verifying
                    self.x = 0.5 * self.x + 0.5 * cand_x
                    self.y = 0.5 * self.y + 0.5 * cand_y
                    self.sigma = max(tracker_sigma, 0.95)
                    self._log_audit(
                        ts,
                        "RECOVERY_VERIFY",
                        "info",
                        f"Controlled recovery: verifying returning fix ({self.recovery_streak}/{self.required_recovery_frames})...",
                    )
                else:
                    self.recovery_verified = True
                    self.ghost_start_ts = None
                    self.x, self.y = cand_x, cand_y
                    self.vx, self.vy = tracker_vel
                    self.sigma = tracker_sigma
                    self.state = "Ambiguous" if engine_out.is_ambiguous else "Tracked"
                    self._log_audit(
                        ts,
                        "RECOVERY_VERIFY",
                        "success",
                        f"COVERED ZONE · new fix verified at ({self.x:.2f}, {self.y:.2f}) -> {self.state}",
                    )
            else:
                # Normal continuous tracking
                self.recovery_verified = True
                self.ghost_start_ts = None
                self.x, self.y = cand_x, cand_y
                self.vx, self.vy = tracker_vel
                self.sigma = tracker_sigma
                self.state = "Ambiguous" if engine_out.is_ambiguous else "Tracked"

        else:
            # Dead Zone: < 2 anchors visible -> Ghost Mode or Lost (USP #05)
            if self.ghost_start_ts is None:
                self.ghost_start_ts = ts

            outage_duration = max(0.0, ts - self.ghost_start_ts)

            if outage_duration >= self.lost_timeout_sec:
                # Prolonged blackout -> State becomes Lost (do not invent a longer fictional path)
                self.state = "Lost"
                self.vx = 0.0
                self.vy = 0.0
                self.sigma = min(4.5, self.sigma + 0.15 * dt)
            else:
                self.state = "Ghost"
                # Dead-reckoning: fuse LSTM momentum with IMU acceleration + yaw rate
                if imu is not None:
                    # Rotate velocity slightly by IMU yaw rate gz
                    dtheta = imu.gz * dt
                    cos_t, sin_t = math.cos(dtheta), math.sin(dtheta)
                    rot_vx = self.vx * cos_t - self.vy * sin_t
                    rot_vy = self.vx * sin_t + self.vy * cos_t
                    # Integrate acceleration with bias-drift damping
                    self.vx = 0.94 * rot_vx + 0.25 * imu.ax * dt
                    self.vy = 0.94 * rot_vy + 0.25 * imu.ay * dt
                else:
                    self.vx *= 0.94
                    self.vy *= 0.94

                cand_x = self.x + self.vx * dt
                cand_y = self.y + self.vy * dt

                # Map-Aware Tracking (USP #07): block impossible moves through walls
                has_los, _ = check_line_of_sight((self.x, self.y), (cand_x, cand_y), self.walls)
                if not has_los:
                    cand_x, cand_y = self.x, self.y
                    self.vx, self.vy = -0.5 * self.vx, -0.5 * self.vy
                else:
                    (cand_x, cand_y), (self.vx, self.vy) = resolve_wall_collision(
                        (cand_x, cand_y),
                        (self.vx, self.vy),
                        self.walls,
                        self.room_width,
                        self.room_height,
                        margin=0.35,
                    )

                self.x = cand_x
                self.y = cand_y

                # Uncertainty widens monotonically with time spent in dead zone
                self.sigma = min(
                    4.0,
                    0.45 + 0.38 * outage_duration + 0.04 * (outage_duration ** 2),
                )

        # Log state transitions (USP #11)
        if self.state != prev_state:
            sev = "info"
            if self.state == "Tracked":
                sev = "success"
            elif self.state in ("Ambiguous", "Ghost"):
                sev = "warn"
            elif self.state == "Lost":
                sev = "danger"
            self._log_audit(
                ts,
                "STATE_CHANGE",
                sev,
                f"State transition: {prev_state} -> {self.state} (anchors_visible={engine_out.anchors_visible}, sigma={self.sigma:.2f}m)",
            )

        # Measurement freshness age (USP #10)
        if self.last_valid_measurement_ts is not None:
            age_ms = max(0.0, (ts - self.last_valid_measurement_ts) * 1000.0)
        else:
            age_ms = 0.0
        # Also include max age among currently online readings
        online_ages = [s.age_ms for s in engine_out.scored_readings if s.online]
        if online_ages and has_coverage:
            age_ms = max(age_ms, max(online_ages))

        mode_w = engine_out.plausible_locations[0].weight if engine_out.plausible_locations else 1.0
        if self.state == "Lost":
            mode_w = 0.25
        conf_2m = compute_confidence_within_radius(self.sigma, radius_m=2.0, mode_weight=mode_w)

        return (
            round(self.x, 3),
            round(self.y, 3),
            round(self.sigma, 3),
            self.state,
            self.recovery_verified,
            conf_2m,
            round(age_ms, 1),
            list(self.audit_log)[:12],
        )
