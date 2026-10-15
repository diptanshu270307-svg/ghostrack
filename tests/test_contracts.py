"""
Unit tests validating data contract schemas according to /docs/contracts.md.
"""

import json
import pytest
from shared.schemas import (
    AnchorReading,
    ReadingsPayload,
    IMUPayload,
    RejectedAnchor,
    EngineOutput,
    WebSocketStreamMessage,
    RoomLayout,
    WallSegment,
    AnchorConfig,
    SetNoiseCommand,
    ToggleAnchorCommand,
    SetNLOSBiasCommand,
    LoadLayoutCommand,
    ResetCommand,
)


def test_readings_payload_schema():
    raw_json = {
        "ts": 1718000000.123,
        "target_id": "target_alpha",
        "readings": [
            {"anchor_id": "A1", "distance": 4.52, "rssi": -62.4, "online": True},
            {"anchor_id": "A2", "distance": 7.89, "rssi": -71.1, "online": True},
            {"anchor_id": "A3", "distance": 3.10, "rssi": None, "online": False},
        ],
    }
    payload = ReadingsPayload.model_validate(raw_json)
    assert payload.target_id == "target_alpha"
    assert len(payload.readings) == 3
    assert payload.readings[0].anchor_id == "A1"
    assert payload.readings[2].online is False
    assert payload.readings[2].rssi is None

    # Check serialization
    dumped = json.loads(payload.model_dump_json())
    assert dumped["readings"][0]["distance"] == 4.52


def test_imu_payload_schema():
    raw_json = {
        "ts": 1718000000.123,
        "ax": 0.052,
        "ay": -0.014,
        "gz": 0.003,
    }
    payload = IMUPayload.model_validate(raw_json)
    assert payload.ax == 0.052
    assert payload.ay == -0.014
    assert payload.gz == 0.003


def test_engine_output_schema():
    raw_json = {
        "ts": 1718000000.123,
        "clean_readings": [
            {"anchor_id": "A1", "distance": 4.52, "rssi": -62.4, "online": True},
            {"anchor_id": "A3", "distance": 3.10, "rssi": -55.8, "online": True},
        ],
        "rejected": [
            {
                "anchor_id": "A2",
                "reason": "geometric_contradiction_wall_bounce_residual_exceeded",
                "residual": 3.25,
            }
        ],
        "anchors_visible": 2,
        "confidence": 0.65,
    }
    output = EngineOutput.model_validate(raw_json)
    assert len(output.clean_readings) == 2
    assert len(output.rejected) == 1
    assert output.rejected[0].anchor_id == "A2"
    assert output.anchors_visible == 2
    assert output.confidence == 0.65


def test_websocket_stream_message_schema():
    raw_json = {
        "ts": 1718000000.123,
        "x": 5.42,
        "y": 3.88,
        "sigma": 0.35,
        "belief_grid": [
            [0.001, 0.002],
            [0.012, 0.098],
        ],
        "grid_shape": [2, 2],
        "mode": "normal",
        "anchors_online": 3,
        "anchors_total": 4,
        "rejected_anchors": ["A2"],
        "ground_truth": {"x": 5.40, "y": 3.85},
    }
    msg = WebSocketStreamMessage.model_validate(raw_json)
    assert msg.mode == "normal"
    assert msg.sigma == 0.35
    assert msg.ground_truth.x == 5.40
    assert msg.rejected_anchors == ["A2"]


def test_command_schemas():
    noise_cmd = SetNoiseCommand.model_validate({"command": "set_noise", "sigma": 0.3})
    assert noise_cmd.sigma == 0.3

    toggle_cmd = ToggleAnchorCommand.model_validate(
        {"command": "toggle_anchor", "anchor_id": "A1", "online": False}
    )
    assert toggle_cmd.anchor_id == "A1"
    assert toggle_cmd.online is False

    nlos_cmd = SetNLOSBiasCommand.model_validate(
        {"command": "set_nlos_bias", "bias": 4.0}
    )
    assert nlos_cmd.bias == 4.0

    reset_cmd = ResetCommand.model_validate({"command": "reset", "seed": 99})
    assert reset_cmd.seed == 99
