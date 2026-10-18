"""
Data contracts and shared Pydantic models for InSight AI / GhostTrack.
Supports all 12 USPs from the Kajukatli presentation:
- 4 Tracking States: Tracked, Ambiguous, Ghost, Lost
- Downweighted liar diagnostics with reliability scores & AI bias estimates
- Multi-modal plausible locations for Ambiguity-Aware Tracking
- Explainable Anchor Recovery Adviser
- Freshness-Aware measurement age tracking
- Auditable Decision Log entries
- Failure Replay & multi-tracker comparison
"""

from typing import List, Optional, Tuple, Literal
from pydantic import BaseModel, Field


TrackingState = Literal["Tracked", "Ambiguous", "Ghost", "Lost"]


class Point2D(BaseModel):
    """2D coordinate in room meters."""
    x: float = Field(..., description="X coordinate in meters")
    y: float = Field(..., description="Y coordinate in meters")
    weight: float = Field(1.0, description="Relative probability weight of this mode")


# ---------------------------------------------------------------------------
# Anchor & Telemetry Schemas
# ---------------------------------------------------------------------------

class AnchorReading(BaseModel):
    """Single anchor distance measurement report."""
    anchor_id: str = Field(..., description="Unique anchor identifier (e.g., 'A1')")
    distance: float = Field(..., description="Observed distance reading in meters")
    rssi: Optional[float] = Field(None, description="Signal strength indicator in dBm, if available")
    online: bool = Field(True, description="Operational status of the anchor")
    ts: Optional[float] = Field(None, description="Individual reading timestamp for freshness checks")


class ReadingsPayload(BaseModel):
    """Payload published over MQTT topic: ghosttrack/room/{room_id}/readings"""
    ts: float = Field(..., description="Epoch timestamp of readings in seconds")
    target_id: str = Field(..., description="Identifier of the tracked entity")
    readings: List[AnchorReading] = Field(..., description="List of anchor observations")
    ground_truth: Optional[Point2D] = Field(
        None,
        description="Optional simulator ground truth for evaluation/demo comparison only (never used by live tracker)",
    )


class IMUPayload(BaseModel):
    """Payload published over MQTT topic: ghosttrack/room/{room_id}/imu"""
    ts: float = Field(..., description="Epoch timestamp of IMU sensor sample")
    ax: float = Field(..., description="Linear acceleration in X (m/s^2)")
    ay: float = Field(..., description="Linear acceleration in Y (m/s^2)")
    gz: float = Field(..., description="Angular velocity around Z yaw axis (rad/s)")


# ---------------------------------------------------------------------------
# Geometric Contradiction Engine & AI Scorer Schemas (USPs #01, #02, #04, #10)
# ---------------------------------------------------------------------------

class RejectedAnchor(BaseModel):
    """Diagnostic detail for an anchor measurement flagged as a liar / downweighted."""
    anchor_id: str = Field(..., description="Anchor flagged as liar")
    reason: str = Field(..., description="Human-readable explanation for distrust")
    residual: float = Field(..., description="Geometric contradiction residual in meters")
    reliability_score: float = Field(0.1, description="Assigned geometric reliability weight [0.0, 1.0]")
    estimated_bias: float = Field(0.0, description="AI-estimated positive NLOS range bias in meters")


class ScoredReading(BaseModel):
    """Anchor reading enriched with geometric reliability weight and AI bias estimate."""
    anchor_id: str
    distance: float
    corrected_distance: float = Field(..., description="Distance after subtracting estimated NLOS bias")
    rssi: Optional[float] = None
    online: bool = True
    is_liar: bool = False
    residual: float = 0.0
    reliability_score: float = Field(1.0, description="Combined geometric + AI reliability weight [0.0, 1.0]")
    ai_score: float = Field(1.0, description="Self-trained AI model confidence score [0.0, 1.0]")
    estimated_bias: float = Field(0.0, description="Self-trained AI estimated NLOS bias in meters")
    reason: str = Field("consistent_los", description="Explanation for the assigned weight")
    age_ms: float = Field(0.0, description="Age of this reading in milliseconds (USP #10)")


class RecoveryAdvice(BaseModel):
    """Explainable recommendation for which anchor to restore next (USP #02)."""
    anchor_id: str = Field(..., description="Offline or degraded anchor ID recommended for restoration")
    priority: int = Field(..., description="1 = highest priority anchor to restore")
    status: Literal["offline", "liar_nlos"] = Field(..., description="Current issue with this anchor")
    reason: str = Field(..., description="Explainable geometric reason why restoring this anchor helps most")
    expected_uncertainty_reduction: float = Field(
        ..., description="Estimated fractional reduction in position uncertainty [0.0, 1.0]"
    )


class AuditLogEntry(BaseModel):
    """Structured audit log entry for state changes, downweights, and advice (USP #11)."""
    ts: float = Field(..., description="Timestamp of decision")
    category: Literal["STATE_CHANGE", "LIAR_DOWNWEIGHT", "RECOVERY_VERIFY", "ADVISER"] = "STATE_CHANGE"
    severity: Literal["info", "warn", "danger", "success"] = "info"
    message: str = Field(..., description="Human-readable auditable explanation")


class EngineOutput(BaseModel):
    """Output structure of the Geometric Contradiction Engine."""
    ts: float = Field(..., description="Timestamp of processed frame")
    clean_readings: List[AnchorReading] = Field(..., description="Readings cleared of severe geometric liars")
    scored_readings: List[ScoredReading] = Field(
        default_factory=list,
        description="All readings kept with reliability scores, residuals, and reasons",
    )
    rejected: List[RejectedAnchor] = Field(default_factory=list, description="List of distrusted/downweighted anchors")
    anchors_visible: int = Field(..., description="Number of valid, non-rejected online anchors")
    confidence: float = Field(1.0, description="Geometric consensus confidence score [0.0, 1.0]")
    consensus_point: Optional[Point2D] = Field(None, description="Weighted least-squares geometric estimate")
    plausible_locations: List[Point2D] = Field(
        default_factory=list,
        description="Candidate locations when geometry is ambiguous (e.g. 2-circle intersections)",
    )
    is_ambiguous: bool = Field(False, description="True when multiple distinct locations fit the readings")


# ---------------------------------------------------------------------------
# WebSocket Communication Schemas
# ---------------------------------------------------------------------------

class GroundTruth(BaseModel):
    """True target location (known to simulator in demo/evaluation mode)."""
    x: float = Field(..., description="True ground truth X coordinate in meters")
    y: float = Field(..., description="True ground truth Y coordinate in meters")


class CompetingTrackerEstimate(BaseModel):
    """Position estimate from a baseline tracker for live comparison & Failure Replay (USP #09)."""
    name: str
    x: Optional[float] = None
    y: Optional[float] = None
    error_m: Optional[float] = None
    rmse_m: float = 0.0
    status: str = "active"


class PeerTargetSummary(BaseModel):
    """Live state of each simultaneously tracked person/tag in the room."""
    target_id: str
    label: str
    color: str
    x: float
    y: float
    true_x: float
    true_y: float
    sigma: float
    confidence_within_2m: float
    state: TrackingState
    rejected_anchors: List[str] = Field(default_factory=list)


class WebSocketStreamMessage(BaseModel):
    """Frame pushed over WebSocket to the React dashboard."""
    ts: float = Field(..., description="Timestamp of estimated frame")
    target_id: str = Field("target_alpha", description="Currently focused target ID")
    active_targets_count: int = Field(1, description="Number of people/tags tracked simultaneously")
    peer_targets: List[PeerTargetSummary] = Field(
        default_factory=list, description="All simultaneously tracked people/robots in the room"
    )
    x: float = Field(..., description="Estimated target X position in meters")
    y: float = Field(..., description="Estimated target Y position in meters")
    sigma: float = Field(..., description="Estimated position standard deviation (uncertainty) in meters")
    confidence_within_2m: float = Field(
        0.95, description="Calibrated probability [0.0, 1.0] of target being within 2.0m (USP #03)"
    )
    belief_grid: List[List[float]] = Field(..., description="2D normalized spatial belief heatmap")
    grid_shape: Tuple[int, int] = Field(..., description="(rows, cols) shape of the belief grid")
    mode: Literal["normal", "ghost"] = Field("normal", description="Legacy mode flag: normal or ghost")
    state: TrackingState = Field(
        "Tracked", description="Explicit PPT state: Tracked, Ambiguous, Ghost, or Lost"
    )
    recovery_verified: bool = Field(
        True, description="True if fix is verified; False during Controlled Recovery check (USP #06)"
    )
    measurement_age_ms: float = Field(
        0.0, description="Age in ms of the latest valid anchor measurement (USP #10)"
    )
    anchors_online: int = Field(..., description="Count of currently active/online anchors")
    anchors_total: int = Field(..., description="Total anchors configured in room")
    rejected_anchors: List[str] = Field(default_factory=list, description="IDs of anchors marked as liars in this frame")
    scored_readings: List[ScoredReading] = Field(
        default_factory=list, description="Per-anchor diagnostics, weights, and AI bias estimates (USP #04)"
    )
    plausible_locations: List[Point2D] = Field(
        default_factory=list, description="Plausible locations for Ambiguity-Aware tracking (USP #01)"
    )
    recovery_advice: List[RecoveryAdvice] = Field(
        default_factory=list, description="Ranked anchor restoration advice (USP #02)"
    )
    audit_events: List[AuditLogEntry] = Field(
        default_factory=list, description="Recent auditable state/weight decisions (USP #11)"
    )
    competing_trackers: List[CompetingTrackerEstimate] = Field(
        default_factory=list, description="Baseline trackers for failure replay & side-by-side comparison (USP #09)"
    )
    ground_truth: Optional[GroundTruth] = Field(None, description="Ground truth location if in demo mode")


# ---------------------------------------------------------------------------
# Room Layout Configuration
# ---------------------------------------------------------------------------

class AnchorConfig(BaseModel):
    """Static coordinates and ID for an anchor in the room."""
    id: str = Field(..., description="Unique anchor ID")
    x: float = Field(..., description="X coordinate in meters")
    y: float = Field(..., description="Y coordinate in meters")
    online: bool = Field(True, description="Default initial state")


class WallSegment(BaseModel):
    """Interior or boundary wall line segment (x1, y1) -> (x2, y2)."""
    x1: float
    y1: float
    x2: float
    y2: float


class RoomLayout(BaseModel):
    """Full floor plan layout definition."""
    room_id: str = Field("room_demo", description="Identifier of the room")
    width: float = Field(12.0, description="Room width in meters (X axis)")
    height: float = Field(10.0, description="Room height in meters (Y axis)")
    anchors: List[AnchorConfig] = Field(default_factory=list, description="Anchor positions")
    walls: List[WallSegment] = Field(default_factory=list, description="Interior wall segments")


# ---------------------------------------------------------------------------
# Dashboard Control Commands (USP #08, #09)
# ---------------------------------------------------------------------------

class SetNoiseCommand(BaseModel):
    command: Literal["set_noise"] = "set_noise"
    sigma: float = Field(..., ge=0.0, description="Gaussian noise standard deviation in meters")


class ToggleAnchorCommand(BaseModel):
    command: Literal["toggle_anchor"] = "toggle_anchor"
    anchor_id: str = Field(..., description="Anchor ID to toggle")
    online: bool = Field(..., description="Desired operational state")


class TriggerBlackoutCommand(BaseModel):
    command: Literal["trigger_blackout"] = "trigger_blackout"
    blackout: bool = Field(True, description="If True, switch all anchors off; if False, restore all")


class SetNLOSBiasCommand(BaseModel):
    command: Literal["set_nlos_bias"] = "set_nlos_bias"
    bias: float = Field(..., ge=0.0, description="Positive distance bias added when wall blocks LOS in meters")


class LoadLayoutCommand(BaseModel):
    command: Literal["load_layout"] = "load_layout"
    layout: RoomLayout = Field(..., description="New layout to apply")


class ResetCommand(BaseModel):
    command: Literal["reset"] = "reset"
    seed: Optional[int] = Field(42, description="Random seed for deterministic replay")


class ReplayCommand(BaseModel):
    command: Literal["start_replay"] = "start_replay"
    scenario: Literal["scripted_chaos", "recorded_session", "live"] = "scripted_chaos"
