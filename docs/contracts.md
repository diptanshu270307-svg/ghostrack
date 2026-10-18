# InSight AI / GhostTrack Data Contracts & 12-USP Communication Protocol

**Team**: Kajukatli (Diptanshu Jamar)  
**Tagline**: *Ambiguity-aware, explainable indoor positioning via geometric outlier rejection and predictive dead-reckoning.*

This document defines the strict communication and data exchange contracts across all components: Simulator, Session Recorder/Replayer, Geometric Contradiction Engine, Self-Trained Reading Scorer, CNN-LSTM Tracker, Ghost Mode Estimator, Anchor Recovery Adviser, FastAPI Backend, and React Dashboard.

---

## 1. MQTT Telemetry

### 1.1 Distance Readings
- **Topic Pattern**: `ghosttrack/room/{room_id}/readings`
- **Direction**: Simulator $\rightarrow$ Broker (Mosquitto / Embedded) $\rightarrow$ Backend
- **Publish Rate**: $\sim 10\text{ Hz}$
- **Payload Schema (`ReadingsPayload`)**:
```json
{
  "ts": 1718000000.123,
  "target_id": "target_alpha",
  "readings": [
    {"anchor_id": "A1", "distance": 4.52, "rssi": -62.4, "online": true, "ts": 1718000000.123},
    {"anchor_id": "A2", "distance": 7.89, "rssi": -71.1, "online": true, "ts": 1718000000.123},
    {"anchor_id": "A3", "distance": 3.10, "rssi": -55.8, "online": true, "ts": 1718000000.123},
    {"anchor_id": "A4", "distance": 9.45, "rssi": null, "online": false, "ts": 1718000000.123}
  ],
  "ground_truth": {"x": 5.40, "y": 3.85, "weight": 1.0}
}
```
*(Note: `ground_truth` is used strictly for dashboard evaluation error display and never reaches live tracker inference.)*

### 1.2 IMU Telemetry (Dead-Reckoning Support)
- **Topic Pattern**: `ghosttrack/room/{room_id}/imu`
- **Payload Schema (`IMUPayload`)**:
```json
{
  "ts": 1718000000.123,
  "ax": 0.052,
  "ay": -0.014,
  "gz": 0.003
}
```

---

## 2. Geometric Contradiction Engine & Self-Trained AI Scorer Output

Instead of discarding suspected NLOS readings blindly, the Geometric Engine downweights "liars" with a continuous reliability score $[0.0, 1.0]$ and keeps the evidence (`scored_readings`) alongside the reason and residual. When only 2 clean anchors intersect inside the room, both intersection points are preserved in `plausible_locations` (`is_ambiguous = true`).

```json
{
  "ts": 1718000000.123,
  "clean_readings": [
    {"anchor_id": "A1", "distance": 4.52, "rssi": -62.4, "online": true},
    {"anchor_id": "A3", "distance": 3.10, "rssi": -55.8, "online": true}
  ],
  "scored_readings": [
    {
      "anchor_id": "A1",
      "distance": 4.52,
      "corrected_distance": 4.48,
      "rssi": -62.4,
      "online": true,
      "is_liar": false,
      "residual": 0.08,
      "reliability_score": 0.96,
      "ai_score": 0.94,
      "estimated_bias": 0.04,
      "reason": "consistent_los",
      "age_ms": 12.4
    },
    {
      "anchor_id": "A2",
      "distance": 7.89,
      "corrected_distance": 5.42,
      "rssi": -71.1,
      "online": true,
      "is_liar": true,
      "residual": 2.47,
      "reliability_score": 0.12,
      "ai_score": 0.15,
      "estimated_bias": 2.47,
      "reason": "NLOS wall-bounce: radius exceeds consensus by +2.47m",
      "age_ms": 12.4
    }
  ],
  "rejected": [
    {
      "anchor_id": "A2",
      "reason": "NLOS wall-bounce: radius exceeds consensus by +2.47m",
      "residual": 2.47,
      "reliability_score": 0.12,
      "estimated_bias": 2.47
    }
  ],
  "anchors_visible": 2,
  "confidence": 0.65,
  "consensus_point": {"x": 5.42, "y": 3.88, "weight": 0.6},
  "plausible_locations": [
    {"x": 5.42, "y": 3.88, "weight": 0.55},
    {"x": 5.42, "y": 6.12, "weight": 0.45}
  ],
  "is_ambiguous": true
}
```

---

## 3. WebSocket Real-Time Stream to Dashboard (`/ws`)

Supports the 4 explicit tracking states (`Tracked`, `Ambiguous`, `Ghost`, `Lost`) and all 12 USPs:

```json
{
  "ts": 1718000000.123,
  "x": 5.42,
  "y": 3.88,
  "sigma": 0.35,
  "confidence_within_2m": 0.88,
  "belief_grid": [[0.001, 0.002], [0.012, 0.098]],
  "grid_shape": [40, 48],
  "mode": "normal",
  "state": "Tracked",
  "recovery_verified": true,
  "measurement_age_ms": 18.5,
  "anchors_online": 3,
  "anchors_total": 4,
  "rejected_anchors": ["A2"],
  "scored_readings": [...],
  "plausible_locations": [{"x": 5.42, "y": 3.88, "weight": 1.0}],
  "recovery_advice": [
    {
      "anchor_id": "A4",
      "priority": 1,
      "status": "offline",
      "reason": "Restoring A4 resolves West-corridor dilution of precision and reduces uncertainty by 58%",
      "expected_uncertainty_reduction": 0.58
    }
  ],
  "audit_events": [
    {
      "ts": 1718000000.123,
      "category": "LIAR_DOWNWEIGHT",
      "severity": "warn",
      "message": "Anchor A2 downweighted (w=0.12): NLOS wall-bounce residual +2.47m"
    }
  ],
  "competing_trackers": [
    {"name": "Naive Trilateration", "x": 7.12, "y": 2.04, "error_m": 2.51, "rmse_m": 1.94, "status": "jumped_nlos"},
    {"name": "Geo Filter + Trilateration", "x": 5.51, "y": 3.95, "error_m": 0.15, "rmse_m": 0.48, "status": "active"},
    {"name": "GhostTrack (Full Pipeline)", "x": 5.42, "y": 3.88, "error_m": 0.04, "rmse_m": 0.22, "status": "Tracked"}
  ],
  "ground_truth": {"x": 5.40, "y": 3.85}
}
```

---

## 4. WebSocket Control Commands from Dashboard

- `{"command": "set_noise", "sigma": 0.25}`
- `{"command": "set_nlos_bias", "bias": 3.5}`
- `{"command": "toggle_anchor", "anchor_id": "A2", "online": false}`
- `{"command": "trigger_blackout", "blackout": true}`
- `{"command": "load_layout", "layout": {...}}`
- `{"command": "reset", "seed": 42}`
- `{"command": "start_replay", "scenario": "scripted_chaos"}`
