"""
Data contracts and shared Pydantic models for InSight AI / GhostTrack.
Used across Simulator, Geometric Engine, CNN-LSTM Tracker, Backend, and Tests.
"""

from typing import List, Optional, Tuple, Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Anchor & Telemetry Schemas
# ---------------------------------------------------------------------------

class AnchorReading(BaseModel):
    """Single anchor distance measurement report."""
    anchor_id: str = Field(..., description="Unique anchor identifier (e.g., 'A1')")
    distance: float = Field(..., description="Observed distance reading in meters")
    rssi: Optional[float] = Field(None, description="Signal strength indicator in dBm, if available")
    online: bool = Field(True, description="Operational status of the anchor")


class ReadingsPayload(BaseModel):
    """Payload published over MQTT topic: ghosttrack/room/{room_id}/readings"""
    ts: float = Field(..., description="Epoch timestamp of readings in seconds")
    target_id: str = Field(..., description="Identifier of the tracked entity")
    readings: List[AnchorReading] = Field(..., description="List of anchor observations")


class IMUPayload(BaseModel):
    """Payload published over MQTT topic: ghosttrack/room/{room_id}/imu"""
    ts: float = Field(..., description="Epoch timestamp of IMU sensor sample")
    ax: float = Field(..., description="Linear acceleration in X (m/s^2)")
    ay: float = Field(..., description="Linear acceleration in Y (m/s^2)")
    gz: float = Field(..., description="Angular velocity around Z yaw axis (rad/s)")


# ---------------------------------------------------------------------------
# Geometric Contradiction Engine Schemas
# ---------------------------------------------------------------------------

class RejectedAnchor(BaseModel):
    """Diagnostic detail for an anchor measurement rejected as a liar."""
    anchor_id: str = Field(..., description="Anchor flagged as liar")
    reason: str = Field(..., description="Reason for rejection (e.g. geometric_contradiction)")
    residual: float = Field(..., description="Contradiction residual metric in meters")


class EngineOutput(BaseModel):
    """Output structure of the Geometric Contradiction Engine."""
    ts: float = Field(..., description="Timestamp of processed frame")
    clean_readings: List[AnchorReading] = Field(..., description="Readings cleared of geometric liars")
    rejected: List[RejectedAnchor] = Field(default_factory=list, description="List of rejected anchor observations")
    anchors_visible: int = Field(..., description="Number of valid, non-rejected online anchors")
    confidence: float = Field(1.0, description="Heuristic geometric confidence score [0.0, 1.0]")


# ---------------------------------------------------------------------------
# WebSocket Communication Schemas
# ---------------------------------------------------------------------------

class GroundTruth(BaseModel):
    """True target location (known to simulator in demo/evaluation mode)."""
    x: float = Field(..., description="True ground truth X coordinate in meters")
    y: float = Field(..., description="True ground truth Y coordinate in meters")


class WebSocketStreamMessage(BaseModel):
    """Frame pushed over WebSocket to the dashboard."""
    ts: float = Field(..., description="Timestamp of estimated frame")
    x: float = Field(..., description="Estimated target X position in meters")
    y: float = Field(..., description="Estimated target Y position in meters")
    sigma: float = Field(..., description="Estimated position standard deviation (uncertainty) in meters")
    belief_grid: List[List[float]] = Field(..., description="2D normalized spatial belief heatmap")
    grid_shape: Tuple[int, int] = Field(..., description="(rows, cols) shape of the belief grid")
    mode: Literal["normal", "ghost"] = Field("normal", description="Tracking mode: normal or ghost dead-reckoning")
    anchors_online: int = Field(..., description="Count of currently active/online anchors")
    anchors_total: int = Field(..., description="Total anchors configured in room")
    rejected_anchors: List[str] = Field(default_factory=list, description="IDs of anchors marked as liars in this frame")
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
# Dashboard Control Commands
# ---------------------------------------------------------------------------

class SetNoiseCommand(BaseModel):
    command: Literal["set_noise"] = "set_noise"
    sigma: float = Field(..., ge=0.0, description="Gaussian noise standard deviation in meters")


class ToggleAnchorCommand(BaseModel):
    command: Literal["toggle_anchor"] = "toggle_anchor"
    anchor_id: str = Field(..., description="Anchor ID to toggle")
    online: bool = Field(..., description="Desired operational state")


class SetNLOSBiasCommand(BaseModel):
    command: Literal["set_nlos_bias"] = "set_nlos_bias"
    bias: float = Field(..., ge=0.0, description="Positive distance bias added when wall blocks LOS in meters")


class LoadLayoutCommand(BaseModel):
    command: Literal["load_layout"] = "load_layout"
    layout: RoomLayout = Field(..., description="New layout to apply")


class ResetCommand(BaseModel):
    command: Literal["reset"] = "reset"
    seed: Optional[int] = Field(42, description="Random seed for deterministic replay")
