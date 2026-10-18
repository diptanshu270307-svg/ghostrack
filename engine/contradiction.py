"""
Zero-Label Geometric Contradiction Engine (USP #01, #04, #10).
Directly integrates the reference Levenberg-Marquardt `multilaterate` and
leave-one-out asymmetric reweighting `geometric_engine` from `ghosttrack.py`,
while exposing full diagnostics, bimodal ambiguity points, and freshness checks
for the live FastAPI + React dashboard.
"""

from typing import List, Dict, Tuple, Optional
import math
import numpy as np

from ghosttrack import multilaterate, geometric_engine
from shared.schemas import (
    Point2D,
    AnchorConfig,
    AnchorReading,
    RejectedAnchor,
    ScoredReading,
    EngineOutput,
)


def circle_intersections(
    x0: float, y0: float, r0: float,
    x1: float, y1: float, r1: float,
) -> List[Tuple[float, float]]:
    """Compute exact intersection points of two circles (x0, y0, r0) and (x1, y1, r1)."""
    dx = x1 - x0
    dy = y1 - y0
    d = math.hypot(dx, dy)
    if d < 1e-6:
        return []
    if d > r0 + r1:
        if d - (r0 + r1) <= 1.0:
            ratio = r0 / max(1e-6, r0 + r1)
            return [(x0 + ratio * dx, y0 + ratio * dy)]
        return []
    if d < abs(r0 - r1):
        if abs(r0 - r1) - d <= 1.0:
            if r0 > r1:
                return [(x0 + (dx / d) * r0, y0 + (dy / d) * r0)]
            else:
                return [(x1 - (dx / d) * r1, y1 - (dy / d) * r1)]
        return []

    a = (r0 * r0 - r1 * r1 + d * d) / (2.0 * d)
    h = math.sqrt(max(0.0, r0 * r0 - a * a))
    x2 = x0 + a * (dx / d)
    y2 = y0 + a * (dy / d)
    rx = -dy * (h / d)
    ry = dx * (h / d)
    pt1 = (x2 + rx, y2 + ry)
    pt2 = (x2 - rx, y2 - ry)
    if h < 1e-3:
        return [pt1]
    return [pt1, pt2]


def naive_trilaterate(
    readings: List[AnchorReading],
    anchor_map: Dict[str, Tuple[float, float]],
    room_size: Tuple[float, float] = (12.0, 10.0),
) -> Optional[Tuple[float, float]]:
    """Plain unweighted Levenberg-Marquardt least-squares multilateration from ghosttrack.py."""
    online = [r for r in readings if r.online and r.anchor_id in anchor_map]
    if len(online) < 3:
        return None
    A = np.array([anchor_map[r.anchor_id] for r in online], dtype=float)
    D = np.array([r.distance for r in online], dtype=float)
    p = multilaterate(A, D)
    return (
        float(np.clip(p[0], 0.0, room_size[0])),
        float(np.clip(p[1], 0.0, room_size[1])),
    )


def weighted_trilaterate(
    coords: np.ndarray,
    dists: np.ndarray,
    weights: np.ndarray,
    room_size: Tuple[float, float] = (12.0, 10.0),
    init_guess: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """Weighted Levenberg-Marquardt multilateration using ghosttrack.multilaterate."""
    p = multilaterate(coords, dists, w=weights, p0=init_guess)
    p_clipped = np.array([
        np.clip(p[0], 0.0, room_size[0]),
        np.clip(p[1], 0.0, room_size[1]),
    ])
    cost = float(np.sum(weights * (np.linalg.norm(p_clipped - coords, axis=1) - dists) ** 2))
    return p_clipped, cost


class GeometricContradictionEngine:
    """
    Wraps `ghosttrack.geometric_engine` (asymmetric NLOS penalty + leave-one-out +
    4-round biweight refinement) and outputs structured diagnostics for the dashboard.
    """

    def __init__(
        self,
        anchors: List[AnchorConfig],
        room_width: float = 12.0,
        room_height: float = 10.0,
        residual_threshold: float = 0.75,
        stale_threshold_ms: float = 600.0,
    ) -> None:
        self.room_width = room_width
        self.room_height = room_height
        self.residual_threshold = residual_threshold
        self.stale_threshold_ms = stale_threshold_ms
        self.anchor_map: Dict[str, Tuple[float, float]] = {
            a.id: (a.x, a.y) for a in anchors
        }

    def update_anchors(
        self,
        anchors: List[AnchorConfig],
        room_width: float,
        room_height: float,
    ) -> None:
        self.room_width = room_width
        self.room_height = room_height
        self.anchor_map = {a.id: (a.x, a.y) for a in anchors}

    def process(
        self,
        ts: float,
        readings: List[AnchorReading],
        prev_pos: Optional[Tuple[float, float]] = None,
    ) -> EngineOutput:
        online_readings = [
            r for r in readings if r.online and r.anchor_id in self.anchor_map
        ]
        scored_list: List[ScoredReading] = []
        rejected_list: List[RejectedAnchor] = []
        clean_list: List[AnchorReading] = []

        for r in readings:
            if not r.online or r.anchor_id not in self.anchor_map:
                scored_list.append(
                    ScoredReading(
                        anchor_id=r.anchor_id,
                        distance=r.distance,
                        corrected_distance=r.distance,
                        rssi=r.rssi,
                        online=False,
                        is_liar=False,
                        residual=0.0,
                        reliability_score=0.0,
                        ai_score=0.0,
                        estimated_bias=0.0,
                        reason="anchor_offline",
                        age_ms=0.0,
                    )
                )

        n_online = len(online_readings)

        if n_online == 0:
            return EngineOutput(
                ts=ts,
                clean_readings=[],
                scored_readings=scored_list,
                rejected=[],
                anchors_visible=0,
                confidence=0.0,
                consensus_point=None,
                plausible_locations=[],
                is_ambiguous=False,
            )

        if n_online == 1:
            r = online_readings[0]
            age_ms = max(0.0, (ts - (r.ts or ts)) * 1000.0)
            fresh_w = math.exp(-max(0.0, age_ms - self.stale_threshold_ms) / 1000.0)
            ax, ay = self.anchor_map[r.anchor_id]
            ref_x, ref_y = prev_pos if prev_pos is not None else (self.room_width / 2.0, self.room_height / 2.0)
            dx, dy = ref_x - ax, ref_y - ay
            dist_ref = max(1e-3, math.hypot(dx, dy))
            est_x = float(np.clip(ax + (dx / dist_ref) * r.distance, 0.0, self.room_width))
            est_y = float(np.clip(ay + (dy / dist_ref) * r.distance, 0.0, self.room_height))

            clean_list.append(r)
            scored_list.append(
                ScoredReading(
                    anchor_id=r.anchor_id,
                    distance=r.distance,
                    corrected_distance=r.distance,
                    rssi=r.rssi,
                    online=True,
                    is_liar=False,
                    residual=0.0,
                    reliability_score=round(0.45 * fresh_w, 3),
                    ai_score=round(0.45 * fresh_w, 3),
                    estimated_bias=0.0,
                    reason="single_anchor_insufficient_geometry",
                    age_ms=round(age_ms, 1),
                )
            )
            pt = Point2D(x=round(est_x, 3), y=round(est_y, 3), weight=1.0)
            return EngineOutput(
                ts=ts,
                clean_readings=clean_list,
                scored_readings=scored_list,
                rejected=[],
                anchors_visible=1,
                confidence=round(0.25 * fresh_w, 3),
                consensus_point=pt,
                plausible_locations=[pt],
                is_ambiguous=True,
            )

        if n_online == 2:
            r0, r1 = online_readings[0], online_readings[1]
            x0, y0 = self.anchor_map[r0.anchor_id]
            x1, y1 = self.anchor_map[r1.anchor_id]
            pts = circle_intersections(x0, y0, r0.distance, x1, y1, r1.distance)

            valid_pts: List[Point2D] = []
            for px, py in pts:
                if -0.5 <= px <= self.room_width + 0.5 and -0.5 <= py <= self.room_height + 0.5:
                    cx = float(np.clip(px, 0.2, self.room_width - 0.2))
                    cy = float(np.clip(py, 0.2, self.room_height - 0.2))
                    valid_pts.append(Point2D(x=round(cx, 3), y=round(cy, 3), weight=0.5))

            if not valid_pts:
                A = np.array([[x0, y0], [x1, y1]], dtype=float)
                D = np.array([r0.distance, r1.distance], dtype=float)
                p_arr = multilaterate(A, D)
                valid_pts = [Point2D(x=round(float(p_arr[0]), 3), y=round(float(p_arr[1]), 3), weight=1.0)]

            if len(valid_pts) == 2 and prev_pos is not None:
                d0 = math.hypot(valid_pts[0].x - prev_pos[0], valid_pts[0].y - prev_pos[1])
                d1 = math.hypot(valid_pts[1].x - prev_pos[0], valid_pts[1].y - prev_pos[1])
                w0, w1 = math.exp(-d0 / 4.0), math.exp(-d1 / 4.0)
                tot = max(1e-6, w0 + w1)
                valid_pts[0].weight = round(w0 / tot, 3)
                valid_pts[1].weight = round(w1 / tot, 3)
                valid_pts.sort(key=lambda p: p.weight, reverse=True)

            is_ambig = len(valid_pts) >= 2 and math.hypot(
                valid_pts[0].x - valid_pts[1].x, valid_pts[0].y - valid_pts[1].y
            ) > 0.8

            for r in online_readings:
                age_ms = max(0.0, (ts - (r.ts or ts)) * 1000.0)
                fresh_w = math.exp(-max(0.0, age_ms - self.stale_threshold_ms) / 1000.0)
                clean_list.append(r)
                scored_list.append(
                    ScoredReading(
                        anchor_id=r.anchor_id,
                        distance=r.distance,
                        corrected_distance=r.distance,
                        rssi=r.rssi,
                        online=True,
                        is_liar=False,
                        residual=0.0,
                        reliability_score=round(0.75 * fresh_w, 3),
                        ai_score=round(0.75 * fresh_w, 3),
                        estimated_bias=0.0,
                        reason="2_anchor_bimodal_ambiguity" if is_ambig else "2_anchor_tangent",
                        age_ms=round(age_ms, 1),
                    )
                )

            return EngineOutput(
                ts=ts,
                clean_readings=clean_list,
                scored_readings=scored_list,
                rejected=[],
                anchors_visible=2,
                confidence=0.52,
                consensus_point=valid_pts[0],
                plausible_locations=valid_pts,
                is_ambiguous=is_ambig,
            )

        # >= 3 anchors online -> Call ghosttrack.geometric_engine directly!
        A = np.array([self.anchor_map[r.anchor_id] for r in online_readings], dtype=float)
        D = np.array([r.distance for r in online_readings], dtype=float)
        p_geo, geo_weights = geometric_engine(A, D, tau=0.6)

        final_xy = np.array([
            float(np.clip(p_geo[0], 0.0, self.room_width)),
            float(np.clip(p_geo[1], 0.0, self.room_height)),
        ])
        pred_d = np.linalg.norm(final_xy - A, axis=1)
        signed_residuals = D - pred_d  # positive = reading is LONGER than geometry (NLOS bounce)

        for idx, r in enumerate(online_readings):
            res_val = float(signed_residuals[idx])
            abs_res = abs(res_val)
            age_ms = max(0.0, (ts - (r.ts or ts)) * 1000.0)
            fresh_w = math.exp(-max(0.0, age_ms - self.stale_threshold_ms) / 1000.0)

            rel_score = float(np.clip(float(geo_weights[idx]) * fresh_w, 0.02, 1.0))
            is_liar = (abs_res > self.residual_threshold) or (float(geo_weights[idx]) < 0.35)
            est_bias = max(0.0, res_val) if res_val > 0.25 else 0.0
            corrected_d = max(0.05, r.distance - est_bias)

            if is_liar:
                if res_val > 0:
                    reason = f"NLOS wall-bounce: radius exceeds consensus by +{res_val:.2f}m"
                else:
                    reason = f"Geometric contradiction: radius shorter than consensus by {res_val:.2f}m"
                rejected_list.append(
                    RejectedAnchor(
                        anchor_id=r.anchor_id,
                        reason=reason,
                        residual=round(abs_res, 3),
                        reliability_score=round(rel_score, 3),
                        estimated_bias=round(est_bias, 3),
                    )
                )
            else:
                reason = "consistent_los"
                if age_ms > self.stale_threshold_ms:
                    reason = f"stale_packet_{int(age_ms)}ms"
                clean_list.append(r)

            scored_list.append(
                ScoredReading(
                    anchor_id=r.anchor_id,
                    distance=r.distance,
                    corrected_distance=round(corrected_d, 3),
                    rssi=r.rssi,
                    online=True,
                    is_liar=is_liar,
                    residual=round(abs_res, 3),
                    reliability_score=round(rel_score, 3),
                    ai_score=round(rel_score, 3),
                    estimated_bias=round(est_bias, 3),
                    reason=reason,
                    age_ms=round(age_ms, 1),
                )
            )

        anchors_visible = len(clean_list)
        consensus_pt = Point2D(
            x=round(float(final_xy[0]), 3),
            y=round(float(final_xy[1]), 3),
            weight=1.0,
        )
        plausible_pts = [consensus_pt]
        is_ambig = False

        if anchors_visible == 2:
            c0, c1 = clean_list[0], clean_list[1]
            x0, y0 = self.anchor_map[c0.anchor_id]
            x1, y1 = self.anchor_map[c1.anchor_id]
            bimodal_pts = circle_intersections(x0, y0, c0.distance, x1, y1, c1.distance)
            cand_modes = []
            for px, py in bimodal_pts:
                if 0.0 <= px <= self.room_width and 0.0 <= py <= self.room_height:
                    cand_modes.append(Point2D(x=round(px, 3), y=round(py, 3), weight=0.5))
            if len(cand_modes) == 2 and math.hypot(
                cand_modes[0].x - cand_modes[1].x, cand_modes[0].y - cand_modes[1].y
            ) > 1.0:
                plausible_pts = cand_modes
                is_ambig = True

        mean_clean_w = float(
            np.mean([s.reliability_score for s in scored_list if s.online and not s.is_liar] or [0.2])
        )
        confidence = float(np.clip(mean_clean_w * min(1.0, anchors_visible / 3.0), 0.1, 0.99))

        return EngineOutput(
            ts=ts,
            clean_readings=clean_list,
            scored_readings=scored_list,
            rejected=rejected_list,
            anchors_visible=anchors_visible,
            confidence=round(confidence, 3),
            consensus_point=consensus_pt,
            plausible_locations=plausible_pts,
            is_ambiguous=is_ambig,
        )
