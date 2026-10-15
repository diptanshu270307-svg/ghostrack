# InSight AI / GhostTrack Data Contracts & Communication Protocol

This document defines the strict communication and data exchange contracts between the components of the InSight AI / GhostTrack system: Simulator, Geometric Engine, CNN-LSTM Tracker, Ghost Mode Estimator, FastAPI Backend, and the React Dashboard.

---

## 1. MQTT Telemetry

### 1.1 Distance Readings
- **Topic Pattern**: `ghosttrack/room/{room_id}/readings`
- **Direction**: Simulator $\rightarrow$ Broker (Mosquitto) $\rightarrow$ Backend
- **Publish Rate**: $\sim 10\text{ Hz}$
- **Payload Schema (JSON)**:
```json
{
  "ts": 1718000000.123,
  "target_id": "target_alpha",
  "readings": [
    {
      "anchor_id": "A1",
      "distance": 4.52,
      "rssi": -62.4,
      "online": true
    },
    {
      "anchor_id": "A2",
      "distance": 7.89,
      "rssi": -71.1,
      "online": true
    },
    {
      "anchor_id": "A3",
      "distance": 3.10,
      "rssi": -55.8,
      "online": true
    },
    {
      "anchor_id": "A4",
      "distance": 9.45,
      "rssi": null,
      "online": false
    }
  ]
}
```

### 1.2 IMU Telemetry (Dead-Reckoning Support)
- **Topic Pattern**: `ghosttrack/room/{room_id}/imu`
- **Direction**: Simulator $\rightarrow$ Broker (Mosquitto) $\rightarrow$ Backend
- **Publish Rate**: $\sim 10\text{ Hz}$ (or synchronized with distance readings)
- **Payload Schema (JSON)**:
```json
{
  "ts": 1718000000.123,
  "ax": 0.052,
  "ay": -0.014,
  "gz": 0.003
}
```
*Note: `ax`, `ay` are linear accelerations in $\text{m/s}^2$ with sensor noise/bias; `gz` is angular yaw rate in $\text{rad/s}$.*

---

## 2. Geometric Contradiction Engine Output

- **Direction**: Engine $\rightarrow$ Tracker / Ghost Mode Pipeline
- **Payload Schema (Dictionary / Pydantic Model)**:
```json
{
  "ts": 1718000000.123,
  "clean_readings": [
    {
      "anchor_id": "A1",
      "distance": 4.52,
      "rssi": -62.4,
      "online": true
    },
    {
      "anchor_id": "A3",
      "distance": 3.10,
      "rssi": -55.8,
      "online": true
    }
  ],
  "rejected": [
    {
      "anchor_id": "A2",
      "reason": "geometric_contradiction_wall_bounce_residual_exceeded",
      "residual": 3.25
    }
  ],
  "anchors_visible": 2,
  "confidence": 0.65
}
```

---

## 3. WebSocket Real-time Stream to Dashboard

- **Endpoint**: `ws://{host}:{port}/ws`
- **Direction**: Backend $\rightarrow$ React Dashboard
- **Payload Schema (JSON)**:
```json
{
  "ts": 1718000000.123,
  "x": 5.42,
  "y": 3.88,
  "sigma": 0.35,
  "belief_grid": [
    [0.001, 0.002, 0.005],
    [0.012, 0.098, 0.015],
    [0.002, 0.004, 0.001]
  ],
  "grid_shape": [50, 50],
  "mode": "normal",
  "anchors_online": 3,
  "anchors_total": 4,
  "rejected_anchors": ["A2"],
  "ground_truth": {
    "x": 5.40,
    "y": 3.85
  }
}
```
*Field definitions:*
- `mode`: `"normal"` when sufficient reliable anchors exist, `"ghost"` when tracking with LSTM momentum + IMU dead-reckoning.
- `sigma`: Standard deviation of spatial uncertainty in meters. In Ghost Mode, `sigma` grows monotonically with time spent without anchor signals.
- `belief_grid`: 2D normalized probability density matrix reflecting current position belief.
- `rejected_anchors`: List of anchor IDs classified as liars / outliers in the latest frame.
- `ground_truth`: Target coordinate known by simulator; null in non-demo deployment.

---

## 4. WebSocket Client Control Commands

Dashboard clients can issue runtime stress-test commands over the same `/ws` connection:

### 4.1 Set Measurement Noise
```json
{
  "command": "set_noise",
  "sigma": 0.25
}
```

### 4.2 Toggle Anchor State
```json
{
  "command": "toggle_anchor",
  "anchor_id": "A2",
  "online": false
}
```

### 4.3 Set NLOS Wall Bounce Bias
```json
{
  "command": "set_nlos_bias",
  "bias": 3.5
}
```

### 4.4 Load Custom Room Layout
```json
{
  "command": "load_layout",
  "layout": {
    "width": 12.0,
    "height": 10.0,
    "anchors": [
      {"id": "A1", "x": 1.0, "y": 1.0},
      {"id": "A2", "x": 11.0, "y": 1.0},
      {"id": "A3", "x": 11.0, "y": 9.0},
      {"id": "A4", "x": 1.0, "y": 9.0}
    ],
    "walls": [
      {"x1": 4.0, "y1": 0.0, "x2": 4.0, "y2": 6.0},
      {"x1": 8.0, "y1": 4.0, "x2": 8.0, "y2": 10.0}
    ]
  }
}
```

### 4.5 Reset Simulation / Tracking
```json
{
  "command": "reset",
  "seed": 42
}
```
