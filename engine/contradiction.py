"""
Zero-Label Geometric Contradiction Engine (USP #01, #04, #10).
Deterministic, label-free outlier detection and downweighting using NumPy & SciPy.
Identifies NLOS wall-bounce 'liar' anchors via circle-intersection consensus,
downweights them with a reliability score while keeping the evidence,
and extracts multi-modal plausible locations when geometry is ambiguous.
"""

from typing import List, Dict, Tuple, Optional
import math
import numpy as np
from scipy.optimize import least_squares

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
    """
    Compute exact intersection points of two circles (x0, y0, r0) and (x1, y1, r1).
    If circles do not strictly intersect due to small noise, returns the closest
    radical midpoint if within tolerance.
    """
    dx = x1 - x0
    dy = y1 - y0
    d = math.hypot(dx, dy)

    if d < 1e-6:
        return []

    # Check if circles are disjoint or one is strictly inside the other
    if d > r0 + r1:
        # If nearly touching (within 1.0m noise), return weighted midpoint along baseline
        if d - (r0 + r1) <= 1.0:
            ratio = r0 / max(1e-6, r0 + r1)
            return [(x0 + ratio * dx, y0 + ratio * dy)]
        return []

    if d < abs(r0 - r1):
        # One circle inside another; if close to tangent, return tangent approximation
        if abs(r0 - r1) - d <= 1.0:
            if r0 > r1:
                return [(x0 + (dx / d) * r0, y0 + (dy / d) * r0)]
            else:
                return [(x1 - (dx / d) * r1, y1 - (dy / d) * r1)]
        return []

    # Standard two-point circle intersection
    a = (r0 * r0 - r1 * r1 + d * d) / (2.0 * d)
    h_sq = max(0.0, r0 * r0 - a * a)
    h = math.sqrt(h_sq)

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
    """
    Standard unweighted least-squares trilateration using ALL online readings
    (including NLOS liars). Used as baseline comparison tracker.
    """
    online = [r for r in readings if r.online and r.anchor_id in anchor_map]
    if len(online) < 2:
        return None

    coords = np.array([anchor_map[r.anchor_id] for r in online], dtype=float)
    dists = np.array([r.distance for r in online], dtype=float)
    init_guess = np.mean(coords, axis=0)

    def residuals(p: np.ndarray) -> np.ndarray:
        return np.hypot(p[0] - coords[:, 0], p[1] - coords[:, 1]) - dists

    res = least_squares(residuals, init_guess, loss="linear")
    x = float(np.clip(res.x[0], 0.0, room_size[0]))
    y = float(np.clip(res.x[1], 0.0, room_size[1]))
    return (x, y)


def weighted_trilaterate(
    coords: np.ndarray,
    dists: np.ndarray,
    weights: np.ndarray,
    room_size: Tuple[float, float] = (12.0, 10.0),
    init_guess: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float]:
    """
    Solve weighted nonlinear least-squares trilateration.
    Returns (estimated_xy, weighted_cost).
    """
    if init_guess is None:
        init_guess = np.average(coords, axis=0, weights=np.maximum(weights, 1e-3))

    sqrt_w = np.sqrt(np.maximum(weights, 1e-4))

    def residuals(p: np.ndarray) -> np.ndarray:
        pred_d = np.hypot(p[0] - coords[:, 0], p[1] - coords[:, 1])
        return sqrt_w * (pred_d - dists)

    bounds = ([0.0, 0.0], [room_size[0], room_size[1]])
    init_clipped = np.clip(init_guess, [0.01, 0.01], [room_size[0] - 0.01, room_size[1] - 0.01])

    res = least_squares(residuals, init_clipped, bounds=bounds, loss="soft_l1", f_scale=0.5)
    return res.x, float(res.cost)


class GeometricContradictionEngine:
    """
    Deterministic, zero-label Geometric Contradiction Engine.
    - Detects NLOS wall-bounce 'liar' readings where circle radii break consensus.
    - Downweights liars with a continuous reliability score [0.0, 1.0] and keeps evidence.
    - Preserves bimodal intersection points when only 2 anchors are reliable (Ambiguous state).
    - Penalizes stale measurements (Freshness-Aware tracking).
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
        """Update anchor coordinates and room dimensions."""
        self.room_width = room_width
        self.room_height = room_height
        self.anchor_map = {a.id: (a.x, a.y) for a in anchors}

    def _find_consensus_point(
        self,
        online_readings: List[AnchorReading],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the geometric consensus point and per-anchor signed residuals
        (observed_distance - consensus_distance).
        Uses Leave-One-Out (LOO) subset consensus + pairwise intersection voting
        so that 1 or 2 massive NLOS wall bounces cannot corrupt the consensus.
        """
        n = len(online_readings)
        coords = np.array([self.anchor_map[r.anchor_id] for r in online_readings], dtype=float)
        dists = np.array([r.distance for r in online_readings], dtype=float)
        room_size = (self.room_width, self.room_height)

        # Gather candidate points from all pairwise circle intersections inside the room
        candidates: List[np.ndarray] = []
        for i in range(n):
            for j in range(i + 1, n):
                pts = circle_intersections(
                    coords[i, 0], coords[i, 1], dists[i],
                    coords[j, 0], coords[j, 1], dists[j],
                )
                for px, py in pts:
                    if -0.5 <= px <= self.room_width + 0.5 and -0.5 <= py <= self.room_height + 0.5:
                        candidates.append(np.array([
                            np.clip(px, 0.0, self.room_width),
                            np.clip(py, 0.0, self.room_height),
                        ]))

        # Also add Leave-One-Out (size 3) trilateration solutions when n >= 4
        if n >= 4:
            for drop_idx in range(n):
                mask = [k for k in range(n) if k != drop_idx]
                sub_c = coords[mask]
                sub_d = dists[mask]
                sub_w = np.ones(len(mask), dtype=float)
                pt, _ = weighted_trilaterate(sub_c, sub_d, sub_w, room_size=room_size)
                candidates.append(pt)

        # Always include full soft_l1 fit as a fallback candidate
        full_pt, _ = weighted_trilaterate(coords, dists, np.ones(n), room_size=room_size)
        candidates.append(full_pt)

        # Score each candidate by truncated inlier error (RANSAC / MLESAC style)
        # Note physical asymmetry: NLOS makes observed d > true_d (positive residual),
        # while observed d << true_d is physically rare unless sensor glitched.
        best_pt = full_pt
        best_score = float("inf")

        # Expected number of inliers is at least 3 when n >= 4, or n when n == 3
        k_inliers = max(3, n - 1) if n >= 4 else n

        for cand in candidates:
            pred_d = np.hypot(cand[0] - coords[:, 0], cand[1] - coords[:, 1])
            signed_res = dists - pred_d
            abs_res = np.abs(signed_res)
            # Sort residuals and sum the best k_inliers residuals
            sorted_idx = np.argsort(abs_res)
            inlier_res = abs_res[sorted_idx[:k_inliers]]
            score = float(np.sum(inlier_res ** 2))
            if score < best_score:
                best_score = score
                best_pt = cand

        # Refine best_pt using only the top k_inliers
        pred_d = np.hypot(best_pt[0] - coords[:, 0], best_pt[1] - coords[:, 1])
        abs_res = np.abs(dists - pred_d)
        inlier_mask = abs_res <= max(self.residual_threshold, float(np.partition(abs_res, min(k_inliers - 1, n - 1))[min(k_inliers - 1, n - 1)]) + 0.1)

        if np.sum(inlier_mask) >= 3:
            refined_pt, _ = weighted_trilaterate(
                coords[inlier_mask],
                dists[inlier_mask],
                np.ones(int(np.sum(inlier_mask))),
                room_size=room_size,
                init_guess=best_pt,
            )
            best_pt = refined_pt

        final_pred_d = np.hypot(best_pt[0] - coords[:, 0], best_pt[1] - coords[:, 1])
        signed_residuals = dists - final_pred_d
        return best_pt, signed_residuals

    def process(
        self,
        ts: float,
        readings: List[AnchorReading],
        prev_pos: Optional[Tuple[float, float]] = None,
    ) -> EngineOutput:
        """
        Run geometric contradiction analysis on a frame of anchor readings.
        Downweights liars, computes reliability scores, and extracts plausible locations.
        """
        online_readings = [
            r for r in readings if r.online and r.anchor_id in self.anchor_map
        ]
        scored_list: List[ScoredReading] = []
        rejected_list: List[RejectedAnchor] = []
        clean_list: List[AnchorReading] = []

        # Record offline anchors in scored_readings with weight 0.0 for full diagnostics
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

        # Case 1: 0 anchors online -> Ghost Mode
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

        # Case 2: 1 anchor online -> Cannot triangulate, low confidence
        if n_online == 1:
            r = online_readings[0]
            age_ms = max(0.0, (ts - (r.ts or ts)) * 1000.0)
            fresh_w = math.exp(-max(0.0, age_ms - self.stale_threshold_ms) / 1000.0)
            ax, ay = self.anchor_map[r.anchor_id]
            # Single circle: project from anchor towards prev_pos or room center
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
                    ai_score=round(0.5 * fresh_w, 3),
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

        # Case 3: 2 anchors online -> Ambiguous bimodal intersection (USP #01)
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
                # Fallback to least-squares midpoint
                coords = np.array([[x0, y0], [x1, y1]], dtype=float)
                dists = np.array([r0.distance, r1.distance], dtype=float)
                pt_arr, _ = weighted_trilaterate(coords, dists, np.ones(2), (self.room_width, self.room_height))
                valid_pts = [Point2D(x=round(float(pt_arr[0]), 3), y=round(float(pt_arr[1]), 3), weight=1.0)]

            # Weight plausible locations slightly by proximity to previous position if available
            if len(valid_pts) == 2 and prev_pos is not None:
                d0 = math.hypot(valid_pts[0].x - prev_pos[0], valid_pts[0].y - prev_pos[1])
                d1 = math.hypot(valid_pts[1].x - prev_pos[0], valid_pts[1].y - prev_pos[1])
                w0 = math.exp(-d0 / 4.0)
                w1 = math.exp(-d1 / 4.0)
                total_w = max(1e-6, w0 + w1)
                valid_pts[0].weight = round(w0 / total_w, 3)
                valid_pts[1].weight = round(w1 / total_w, 3)
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

        # Case 4: >= 3 anchors online -> Full Geometric Contradiction Rejection
        consensus_xy, signed_residuals = self._find_consensus_point(online_readings)

        weights = []
        for idx, r in enumerate(online_readings):
            res_val = float(signed_residuals[idx])
            abs_res = abs(res_val)
            age_ms = max(0.0, (ts - (r.ts or ts)) * 1000.0)
            fresh_w = math.exp(-max(0.0, age_ms - self.stale_threshold_ms) / 1000.0)

            # Continuous reliability weight: Gaussian roll-off with residual
            raw_w = math.exp(-((max(0.0, abs_res - 0.15) / 0.65) ** 2))
            rel_score = float(np.clip(raw_w * fresh_w, 0.04, 1.0))

            is_liar = abs_res > self.residual_threshold
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

            weights.append(rel_score)
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

        # Re-solve final weighted consensus using continuous reliability weights
        coords = np.array([self.anchor_map[r.anchor_id] for r in online_readings], dtype=float)
        dists = np.array([r.distance for r in online_readings], dtype=float)
        w_arr = np.array(weights, dtype=float)
        final_xy, _ = weighted_trilaterate(
            coords, dists, w_arr, (self.room_width, self.room_height), init_guess=consensus_xy
        )

        anchors_visible = len(clean_list)
        plausible_pts: List[Point2D] = []
        is_ambig = False

        # If rejecting liars left only 2 clean anchors, check if those 2 clean anchors have bimodal ambiguity!
        if anchors_visible == 2:
            c0, c1 = clean_list[0], clean_list[1]
            x0, y0 = self.anchor_map[c0.anchor_id]
            x1, y1 = self.anchor_map[c1.anchor_id]
            bimodal_pts = circle_intersections(x0, y0, c0.distance, x1, y1, c1.distance)
            for px, py in bimodal_pts:
                if 0.0 <= px <= self.room_width and 0.0 <= py <= self.room_height:
                    # Use the downweighted liar's upper-bound radius (true_d <= liar_d) to weight modes
                    mode_w = 0.5
                    for rej in rejected_list:
                        lx, ly = self.anchor_map[rej.anchor_id]
                        liar_r = next(r.distance for r in online_readings if r.anchor_id == rej.anchor_id)
                        dist_to_liar = math.hypot(px - lx, py - ly)
                        # Physical NLOS constraint: true distance cannot exceed bounced NLOS distance
                        if dist_to_liar <= liar_r + 0.5:
                            mode_w += 0.25
                    plausible_pts.append(Point2D(x=round(px, 3), y=round(py, 3), weight=round(mode_w, 3)))

            if len(plausible_pts) == 2:
                sep = math.hypot(
                    plausible_pts[0].x - plausible_pts[1].x,
                    plausible_pts[0].y - plausible_pts[1].y,
                )
                if sep > 1.0:
                    total_mw = sum(p.weight for p in plausible_pts)
                    for p in plausible_pts:
                        p.weight = round(p.weight / max(1e-6, total_mw), 3)
                    plausible_pts.sort(key=lambda p: p.weight, reverse=True)
                    is_ambig = True
                    final_xy = np.array([plausible_pts[0].x, plausible_pts[0].y])

        consensus_pt = Point2D(
            x=round(float(final_xy[0]), 3),
            y=round(float(final_xy[1]), 3),
            weight=1.0,
        )
        if not plausible_pts:
            plausible_pts = [consensus_pt]

        # Compute overall geometric confidence [0.0, 1.0]
        mean_clean_w = float(np.mean([s.reliability_score for s in scored_list if s.online and not s.is_liar] or [0.2]))
        count_factor = min(1.0, anchors_visible / 3.0)
        confidence = float(np.clip(mean_clean_w * count_factor, 0.1, 0.99))

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
