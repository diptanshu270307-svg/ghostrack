"""
Explainable Anchor Recovery Adviser (USP #02).
Ranks offline and NLOS-degraded anchors by computing how much restoring each
anchor would reduce geometric dilution of precision (GDOP) or resolve
bimodal corridor ambiguity.
"""

from typing import List, Tuple
import math

from shared.schemas import (
    RoomLayout,
    EngineOutput,
    RecoveryAdvice,
)
from simulator.geometry import check_line_of_sight


def rank_recovery_anchors(
    layout: RoomLayout,
    engine_out: EngineOutput,
    estimated_pos: Tuple[float, float],
) -> List[RecoveryAdvice]:
    """
    Evaluate all offline or distrusted anchors and rank which anchor to restore
    first, along with a human-readable geometric explanation.
    """
    tx, ty = estimated_pos
    anchor_map = {a.id: (a.x, a.y) for a in layout.anchors}

    # Active clean anchors and their unit sightline vectors from target
    clean_ids = {r.anchor_id for r in engine_out.clean_readings}
    clean_vectors: List[Tuple[float, float]] = []
    for cid in clean_ids:
        if cid in anchor_map:
            ax, ay = anchor_map[cid]
            d = max(1e-3, math.hypot(ax - tx, ay - ty))
            clean_vectors.append(((ax - tx) / d, (ay - ty) / d))

    candidates: List[ Tuple[float, str, str, str] ] = []

    for sr in engine_out.scored_readings:
        if sr.online and not sr.is_liar:
            continue  # Already healthy and trusted

        aid = sr.anchor_id
        if aid not in anchor_map:
            continue

        ax, ay = anchor_map[aid]
        dist_to_target = max(0.1, math.hypot(ax - tx, ay - ty))
        ux, uy = (ax - tx) / dist_to_target, (ay - ty) / dist_to_target

        has_los, wall_cuts = check_line_of_sight((tx, ty), (ax, ay), layout.walls)
        status = "offline" if not sr.online else "liar_nlos"

        # 1. Ambiguity-breaking score (when 2 plausible locations exist)
        ambiguity_gain = 0.0
        if len(engine_out.plausible_locations) >= 2:
            p1 = engine_out.plausible_locations[0]
            p2 = engine_out.plausible_locations[1]
            d1 = math.hypot(ax - p1.x, ay - p1.y)
            d2 = math.hypot(ax - p2.x, ay - p2.y)
            # Larger distance difference between the two modes = stronger disambiguation
            ambiguity_gain = min(1.0, abs(d1 - d2) / 4.0)

        # 2. Angular diversity / GDOP improvement score
        if not clean_vectors:
            angular_gain = 0.85
        else:
            # Prefer sightlines orthogonal to existing clean sightlines (min |cos(theta)|)
            max_align = max(abs(ux * vx + uy * vy) for vx, vy in clean_vectors)
            angular_gain = 1.0 - 0.6 * max_align

        # 3. Proximity & LOS bonus
        los_factor = 1.0 if has_los else max(0.35, 0.75 - 0.15 * wall_cuts)
        proximity_factor = max(0.3, 1.0 - (dist_to_target / math.hypot(layout.width, layout.height)) * 0.5)

        raw_utility = (0.45 * angular_gain + 0.35 * ambiguity_gain + 0.20 * proximity_factor) * los_factor
        # Offline anchors have slightly higher actionability than active NLOS anchors
        if status == "offline":
            raw_utility = min(0.95, raw_utility * 1.15)

        reduction = round(max(0.15, min(0.88, raw_utility)), 2)

        # Construct explainable reason
        reasons = []
        if ambiguity_gain > 0.35 and len(engine_out.plausible_locations) >= 2:
            p1, p2 = engine_out.plausible_locations[0], engine_out.plausible_locations[1]
            reasons.append(
                f"breaks bimodal ambiguity between ({p1.x:.1f}, {p1.y:.1f}) and ({p2.x:.1f}, {p2.y:.1f})"
            )
        if len(clean_vectors) == 0:
            reasons.append("restores primary range fix to exit Ghost/Lost blackout")
        elif angular_gain > 0.60:
            reasons.append("provides orthogonal sightline geometry (lowers GDOP)")
        else:
            reasons.append("adds redundant trilateration constraint")

        if has_los:
            reasons.append(f"has clear line-of-sight ({dist_to_target:.1f}m)")
        else:
            reasons.append(f"mitigates {wall_cuts}-wall bounce if repositioned/cleared")

        reason_str = f"Restore {aid}: " + ", ".join(reasons) + f" (-{int(reduction * 100)}% uncertainty)."
        candidates.append((reduction, aid, status, reason_str))

    candidates.sort(key=lambda item: item[0], reverse=True)

    advice_list: List[RecoveryAdvice] = []
    for idx, (red, aid, status, r_str) in enumerate(candidates):
        advice_list.append(
            RecoveryAdvice(
                anchor_id=aid,
                priority=idx + 1,
                status=status,  # type: ignore[arg-type]
                reason=r_str,
                expected_uncertainty_reduction=red,
            )
        )

    return advice_list
