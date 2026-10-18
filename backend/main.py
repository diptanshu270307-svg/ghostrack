"""
FastAPI Backend & Real-Time WebSocket Streaming Server for InSight AI / GhostTrack.
Integrates:
  - MQTT Telemetry Ingestion (with seamless internal simulator fallback)
  - Zero-Label Geometric Contradiction Engine
  - Self-Trained AI Reading Scorer & Bias Estimator
  - CNN-LSTM Sequential Tracker
  - 4-State Ghost Mode Dead-Reckoning Controller (Tracked, Ambiguous, Ghost, Lost)
  - Explainable Anchor Recovery Adviser
  - Multi-Tracker Failure Replay & Live RMSE Comparison
  - Map-Aware Multi-Modal 2D Belief Heatmap Generator
"""

from contextlib import asynccontextmanager
from typing import Set, List, Optional, Tuple
from collections import deque
import asyncio
import json
import math
import os
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import paho.mqtt.client as mqtt

from shared.schemas import (
    RoomLayout,
    ReadingsPayload,
    IMUPayload,
    GroundTruth,
    CompetingTrackerEstimate,
    WebSocketStreamMessage,
)
from simulator.sim import SensorSimulator, get_default_layout
from engine.contradiction import GeometricContradictionEngine, naive_trilaterate
from engine.adviser import rank_recovery_anchors
from model.scorer import ReadingScorer
from model.tracker import CNNLSTMTracker
from model.ghost import GhostModeController
from backend.belief import compute_belief_grid


class GhostTrackPipelineService:
    """
    End-to-end stateful pipeline orchestrating all stages of GhostTrack
    and managing connected WebSocket clients.
    """

    def __init__(self) -> None:
        self.mqtt_host = os.getenv("MQTT_HOST", "127.0.0.1")
        self.mqtt_port = int(os.getenv("MQTT_PORT", "1883"))
        self.room_id = os.getenv("MQTT_ROOM_ID", "room_demo")

        self.layout: RoomLayout = get_default_layout()
        self.layout.room_id = self.room_id

        # Internal simulator (used as fallback or primary when no external MQTT publisher is active)
        self.sim = SensorSimulator(layout=self.layout, rate_hz=10.0, seed=42)

        # Pipeline stages
        self.engine = GeometricContradictionEngine(
            anchors=self.layout.anchors,
            room_width=self.layout.width,
            room_height=self.layout.height,
        )
        self.scorer = ReadingScorer(weights_path="model/weights/reading_scorer.pt")
        self.tracker = CNNLSTMTracker(
            room_width=self.layout.width,
            room_height=self.layout.height,
            weights_path="model/weights/cnn_lstm_tracker.pt",
            anchor_map=self.engine.anchor_map,
        )
        self.ghost = GhostModeController(
            room_width=self.layout.width,
            room_height=self.layout.height,
            walls=self.layout.walls,
        )

        # Latest telemetry received over MQTT
        self.latest_mqtt_readings: Optional[ReadingsPayload] = None
        self.latest_mqtt_imu: Optional[IMUPayload] = None
        self.last_mqtt_rx_time: float = 0.0
        self.mqtt_connected: bool = False
        self.mqtt_client: Optional[mqtt.Client] = None

        # Competing trackers state for USP #09 (Failure Replay & Live Comparison)
        self.last_naive_xy: Tuple[float, float] = (self.layout.width / 2.0, self.layout.height / 2.0)
        self.last_geo_xy: Tuple[float, float] = (self.layout.width / 2.0, self.layout.height / 2.0)
        self.naive_sq_errors: deque = deque(maxlen=200)
        self.geo_sq_errors: deque = deque(maxlen=200)
        self.gt_sq_errors: deque = deque(maxlen=200)

        self.latest_ws_message: Optional[WebSocketStreamMessage] = None
        self.clients: Set[WebSocket] = set()

    def start_mqtt(self) -> None:
        """Connect to MQTT broker if available; degrade gracefully if broker is absent."""
        try:
            client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id=f"ghosttrack_backend_{int(time.time() * 1000) % 100000}",
            )

            def on_connect(c, userdata, flags, rc, properties=None):
                if rc == 0:
                    self.mqtt_connected = True
                    c.subscribe("ghosttrack/room/+/readings")
                    c.subscribe("ghosttrack/room/+/imu")

            def on_disconnect(c, userdata, flags, rc, properties=None):
                self.mqtt_connected = False

            def on_message(c, userdata, msg):
                try:
                    data = json.loads(msg.payload.decode("utf-8"))
                    if msg.topic.endswith("/readings"):
                        self.latest_mqtt_readings = ReadingsPayload.model_validate(data)
                        self.last_mqtt_rx_time = time.time()
                    elif msg.topic.endswith("/imu"):
                        self.latest_mqtt_imu = IMUPayload.model_validate(data)
                except Exception:
                    pass

            client.on_connect = on_connect
            client.on_disconnect = on_disconnect
            client.on_message = on_message
            client.connect_async(self.mqtt_host, self.mqtt_port, keepalive=30)
            client.loop_start()
            self.mqtt_client = client
        except Exception:
            self.mqtt_connected = False

    def stop_mqtt(self) -> None:
        if self.mqtt_client is not None:
            try:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
            except Exception:
                pass

    def apply_control_command(self, cmd_data: dict) -> None:
        """Apply control command to internal simulator, pipeline, and forward over MQTT."""
        cmd_type = cmd_data.get("command")
        if cmd_type == "set_noise":
            self.sim.set_noise(float(cmd_data.get("sigma", self.sim.noise_sigma)))
        elif cmd_type == "set_nlos_bias":
            self.sim.set_nlos_bias(float(cmd_data.get("bias", self.sim.nlos_bias)))
        elif cmd_type == "toggle_anchor":
            aid = str(cmd_data.get("anchor_id", ""))
            online = bool(cmd_data.get("online", True))
            self.sim.toggle_anchor(aid, online)
            for a in self.layout.anchors:
                if a.id == aid:
                    a.online = online
        elif cmd_type == "trigger_blackout":
            blackout = bool(cmd_data.get("blackout", True))
            self.sim.trigger_blackout(blackout)
            for a in self.layout.anchors:
                a.online = not blackout
        elif cmd_type == "reset":
            seed = int(cmd_data.get("seed", 42))
            self.sim.reset(seed)
            self.tracker.reset()
            self.ghost.reset()
            self.naive_sq_errors.clear()
            self.geo_sq_errors.clear()
            self.gt_sq_errors.clear()
        elif cmd_type == "start_replay":
            self.sim.start_scripted_chaos()
            self.tracker.reset()
            self.ghost.reset()
            self.naive_sq_errors.clear()
            self.geo_sq_errors.clear()
            self.gt_sq_errors.clear()
        elif cmd_type == "load_layout":
            new_layout = RoomLayout.model_validate(cmd_data.get("layout"))
            self.update_layout(new_layout)

        # Forward command over MQTT for external simulator synchronization
        if self.mqtt_client and self.mqtt_connected:
            try:
                control_topic = f"ghosttrack/room/{self.layout.room_id}/control"
                self.mqtt_client.publish(control_topic, json.dumps(cmd_data), qos=0)
            except Exception:
                pass

    def update_layout(self, new_layout: RoomLayout) -> None:
        self.layout = new_layout
        self.sim.load_layout(new_layout)
        self.engine.update_anchors(new_layout.anchors, new_layout.width, new_layout.height)
        self.tracker.update_room_size(new_layout.width, new_layout.height, self.engine.anchor_map)
        self.ghost.update_layout(new_layout.width, new_layout.height, new_layout.walls)

    def process_frame(
        self,
        readings_payload: ReadingsPayload,
        imu_payload: Optional[IMUPayload],
        ground_truth_xy: Optional[Tuple[float, float]] = None,
    ) -> WebSocketStreamMessage:
        """Run a single telemetry frame through all 4 stages and build WebSocket message."""
        ts = readings_payload.ts

        # Stage 1: Geometric Contradiction Engine
        eng_out = self.engine.process(
            ts=ts,
            readings=readings_payload.readings,
            prev_pos=(self.ghost.x, self.ghost.y),
        )

        # Stage 2: Self-Trained AI Reading Scorer & Bias Estimator
        eng_out.scored_readings = self.scorer.score_and_correct(eng_out.scored_readings)

        # Stage 3: CNN-LSTM Sequential Tracker
        t_x, t_y, t_sigma, t_vel = self.tracker.step(
            eng_out, imu_payload, total_anchors=len(self.layout.anchors)
        )

        # Explainable Anchor Recovery Adviser (USP #02)
        recovery_advice = rank_recovery_anchors(
            self.layout, eng_out, (t_x, t_y)
        )
        top_advice_id = recovery_advice[0].anchor_id if recovery_advice else None

        # Stage 4: Ghost Mode & 4-State Lifecycle Controller
        gx, gy, g_sigma, state, recovery_verified, conf_2m, age_ms, audit_events = self.ghost.step(
            ts=ts,
            engine_out=eng_out,
            tracker_xy=(t_x, t_y),
            tracker_sigma=t_sigma,
            tracker_vel=t_vel,
            imu=imu_payload,
            top_advice_anchor=top_advice_id,
        )

        # Stage 5: Map-Aware Multi-Modal 2D Belief Heatmap
        modes_for_heatmap = (
            eng_out.plausible_locations
            if state == "Ambiguous" and len(eng_out.plausible_locations) >= 2
            else None
        )
        belief_grid, grid_shape = compute_belief_grid(
            x=gx,
            y=gy,
            sigma=g_sigma,
            room_width=self.layout.width,
            room_height=self.layout.height,
            plausible_locations=modes_for_heatmap,
            walls=self.layout.walls,
        )

        # Competing Trackers Comparison (USP #09)
        naive_xy = naive_trilaterate(
            readings_payload.readings,
            self.engine.anchor_map,
            (self.layout.width, self.layout.height),
        )
        if naive_xy is not None:
            self.last_naive_xy = naive_xy
            naive_status = "jumped_nlos" if len(eng_out.rejected) > 0 else "active"
        else:
            naive_status = "frozen_blindspot"

        if eng_out.consensus_point is not None:
            self.last_geo_xy = (eng_out.consensus_point.x, eng_out.consensus_point.y)
            geo_status = "ambiguous" if eng_out.is_ambiguous else "active"
        else:
            geo_status = "frozen_blindspot"

        gt_obj: Optional[GroundTruth] = None
        err_naive = err_geo = err_gt = None

        if ground_truth_xy is not None:
            tx, ty = ground_truth_xy
            gt_obj = GroundTruth(x=round(tx, 3), y=round(ty, 3))
            err_naive = round(math.hypot(self.last_naive_xy[0] - tx, self.last_naive_xy[1] - ty), 3)
            err_geo = round(math.hypot(self.last_geo_xy[0] - tx, self.last_geo_xy[1] - ty), 3)
            err_gt = round(math.hypot(gx - tx, gy - ty), 3)

            self.naive_sq_errors.append(err_naive ** 2)
            self.geo_sq_errors.append(err_geo ** 2)
            self.gt_sq_errors.append(err_gt ** 2)

        def calc_rmse(buf: deque) -> float:
            if not buf:
                return 0.0
            return round(math.sqrt(sum(buf) / len(buf)), 3)

        competing = [
            CompetingTrackerEstimate(
                name="Naive Trilateration",
                x=round(self.last_naive_xy[0], 3),
                y=round(self.last_naive_xy[1], 3),
                error_m=err_naive,
                rmse_m=calc_rmse(self.naive_sq_errors),
                status=naive_status,
            ),
            CompetingTrackerEstimate(
                name="Geo Filter + Trilateration",
                x=round(self.last_geo_xy[0], 3),
                y=round(self.last_geo_xy[1], 3),
                error_m=err_geo,
                rmse_m=calc_rmse(self.geo_sq_errors),
                status=geo_status,
            ),
            CompetingTrackerEstimate(
                name="GhostTrack (Full Pipeline)",
                x=gx,
                y=gy,
                error_m=err_gt,
                rmse_m=calc_rmse(self.gt_sq_errors),
                status=state,
            ),
        ]

        anchors_online = sum(1 for r in readings_payload.readings if r.online)
        legacy_mode = "ghost" if state in ("Ghost", "Lost") else "normal"

        msg = WebSocketStreamMessage(
            ts=round(ts, 3),
            x=gx,
            y=gy,
            sigma=g_sigma,
            confidence_within_2m=conf_2m,
            belief_grid=belief_grid,
            grid_shape=grid_shape,
            mode=legacy_mode,  # type: ignore[arg-type]
            state=state,
            recovery_verified=recovery_verified,
            measurement_age_ms=age_ms,
            anchors_online=anchors_online,
            anchors_total=len(self.layout.anchors),
            rejected_anchors=[r.anchor_id for r in eng_out.rejected],
            scored_readings=eng_out.scored_readings,
            plausible_locations=eng_out.plausible_locations,
            recovery_advice=recovery_advice,
            audit_events=audit_events,
            competing_trackers=competing,
            ground_truth=gt_obj,
        )
        self.latest_ws_message = msg
        return msg

    def tick(self) -> WebSocketStreamMessage:
        """Advance pipeline using external MQTT frame if fresh, else internal simulator."""
        now = time.time()
        if (
            self.latest_mqtt_readings is not None
            and (now - self.last_mqtt_rx_time) < 0.45
            and not self.sim.scripted_mode
        ):
            rp = self.latest_mqtt_readings
            imu = self.latest_mqtt_imu
            gt_xy = (
                (rp.ground_truth.x, rp.ground_truth.y)
                if rp.ground_truth is not None
                else None
            )
            return self.process_frame(rp, imu, ground_truth_xy=gt_xy)

        rp, imu, gt_xy, _ = self.sim.step()
        return self.process_frame(rp, imu, ground_truth_xy=gt_xy)


pipeline_service = GhostTrackPipelineService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline_service.start_mqtt()
    stop_flag = False

    async def broadcast_loop():
        while not stop_flag:
            t0 = time.time()
            msg = pipeline_service.tick()
            if pipeline_service.clients:
                payload_str = msg.model_dump_json()
                stale_clients: List[WebSocket] = []
                for ws in list(pipeline_service.clients):
                    try:
                        await asyncio.wait_for(ws.send_text(payload_str), timeout=0.08)
                    except Exception:
                        stale_clients.append(ws)
                for ws in stale_clients:
                    pipeline_service.clients.discard(ws)

            elapsed = time.time() - t0
            await asyncio.sleep(max(0.01, 0.10 - elapsed))

    task = asyncio.create_task(broadcast_loop())
    yield
    stop_flag = True
    task.cancel()
    pipeline_service.stop_mqtt()


app = FastAPI(
    title="InSight AI / GhostTrack API",
    description="Ambiguity-aware, explainable indoor positioning via geometric outlier rejection and predictive dead-reckoning.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def get_health():
    return {
        "status": "ok",
        "service": "InSight AI / GhostTrack",
        "team": "Kajukatli (Diptanshu Jamar)",
        "mqtt_connected": pipeline_service.mqtt_connected,
        "state": pipeline_service.ghost.state,
        "scorer_weights_loaded": pipeline_service.scorer.trained_loaded,
        "tracker_weights_loaded": pipeline_service.tracker.trained_loaded,
        "clients_connected": len(pipeline_service.clients),
    }


@app.get("/config")
async def get_config():
    return {
        "layout": pipeline_service.layout.model_dump(),
        "noise_sigma": pipeline_service.sim.noise_sigma,
        "nlos_bias": pipeline_service.sim.nlos_bias,
        "anchor_status": pipeline_service.sim.anchor_status,
        "scripted_mode": pipeline_service.sim.scripted_mode,
    }


@app.get("/metrics")
async def get_metrics():
    report_path = "data/evaluation_report.json"
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"status": "not_generated_yet"}


@app.post("/layout")
async def post_layout(layout: RoomLayout):
    pipeline_service.update_layout(layout)
    return {"status": "updated", "layout": pipeline_service.layout.model_dump()}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    pipeline_service.clients.add(websocket)
    try:
        if pipeline_service.latest_ws_message is not None:
            await websocket.send_text(pipeline_service.latest_ws_message.model_dump_json())
        while True:
            raw_text = await websocket.receive_text()
            try:
                cmd_data = json.loads(raw_text)
                pipeline_service.apply_control_command(cmd_data)
            except Exception:
                pass
    except WebSocketDisconnect:
        pipeline_service.clients.discard(websocket)
    except Exception:
        pipeline_service.clients.discard(websocket)
