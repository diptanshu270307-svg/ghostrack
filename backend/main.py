"""
FastAPI Backend & Real-Time WebSocket Streaming Server for InSight AI / GhostTrack.
Supports Simultaneous Multi-Person / Multi-Tag Tracking in the same room:
  - Up to 3 concurrent people/robots (Person Alpha, Robot Bravo, Staff Charlie),
    each with its own independent Geometric Contradiction check, CNN-LSTM momentum memory,
    and 4-State Ghost Mode controller (Tracked, Ambiguous, Ghost, Lost).
"""

from contextlib import asynccontextmanager
from typing import Dict, Set, List, Optional, Tuple
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
    PeerTargetSummary,
    WebSocketStreamMessage,
)
from simulator.sim import SensorSimulator, get_default_layout
from simulator.trajectory import TrajectoryGenerator
from engine.contradiction import GeometricContradictionEngine, naive_trilaterate
from engine.adviser import rank_recovery_anchors
from model.scorer import ReadingScorer
from model.tracker import CNNLSTMTracker
from model.ghost import GhostModeController
from backend.belief import compute_belief_grid

TARGET_PROFILES = [
    {"id": "target_alpha", "label": "Person Alpha", "color": "#10b981", "seed": 42, "speed": 1.35},
    {"id": "target_bravo", "label": "Robot Bravo", "color": "#38bdf8", "seed": 137, "speed": 1.55},
    {"id": "target_charlie", "label": "Staff Charlie", "color": "#f472b6", "seed": 271, "speed": 1.15},
]


class TargetTrackInstance:
    """Independent per-person tracking state (Trajectory + Scorer + CNN-LSTM + Ghost Mode)."""

    def __init__(self, profile: dict, layout: RoomLayout, anchor_map: Dict[str, Tuple[float, float]]) -> None:
        self.target_id: str = profile["id"]
        self.label: str = profile["label"]
        self.color: str = profile["color"]
        self.seed: int = profile["seed"]
        self.speed: float = profile["speed"]

        self.trajectory = TrajectoryGenerator(
            room_width=layout.width,
            room_height=layout.height,
            walls=layout.walls,
            nominal_speed=self.speed,
            seed=self.seed,
        )
        # Offset initial positions so multiple people start in different parts of the room
        if self.target_id == "target_bravo":
            self.trajectory.x = layout.width * 0.72
            self.trajectory.y = layout.height * 0.28
        elif self.target_id == "target_charlie":
            self.trajectory.x = layout.width * 0.68
            self.trajectory.y = layout.height * 0.75

        self.scorer = ReadingScorer(weights_path="model/weights/reading_scorer.pt")
        self.tracker = CNNLSTMTracker(
            room_width=layout.width,
            room_height=layout.height,
            weights_path="model/weights/cnn_lstm_tracker.pt",
            anchor_map=anchor_map,
        )
        self.ghost = GhostModeController(
            room_width=layout.width,
            room_height=layout.height,
            walls=layout.walls,
        )
        self.ghost.x = self.trajectory.x
        self.ghost.y = self.trajectory.y

    def reset(self, layout: RoomLayout) -> None:
        self.trajectory.reset(self.seed)
        if self.target_id == "target_bravo":
            self.trajectory.x = layout.width * 0.72
            self.trajectory.y = layout.height * 0.28
        elif self.target_id == "target_charlie":
            self.trajectory.x = layout.width * 0.68
            self.trajectory.y = layout.height * 0.75
        self.tracker.reset()
        self.ghost.reset()
        self.ghost.x = self.trajectory.x
        self.ghost.y = self.trajectory.y

    def update_layout(self, layout: RoomLayout, anchor_map: Dict[str, Tuple[float, float]]) -> None:
        self.trajectory.update_layout(layout.width, layout.height, layout.walls)
        self.tracker.update_room_size(layout.width, layout.height, anchor_map)
        self.ghost.update_layout(layout.width, layout.height, layout.walls)


class GhostTrackPipelineService:
    """
    End-to-end stateful pipeline orchestrating all stages of GhostTrack
    for 1 to 3 simultaneous people/tags in the room.
    """

    def __init__(self) -> None:
        self.mqtt_host = os.getenv("MQTT_HOST", "127.0.0.1")
        self.mqtt_port = int(os.getenv("MQTT_PORT", "1883"))
        self.room_id = os.getenv("MQTT_ROOM_ID", "room_demo")

        self.layout: RoomLayout = get_default_layout()
        self.layout.room_id = self.room_id

        self.sim = SensorSimulator(layout=self.layout, rate_hz=10.0, seed=42)
        self.engine = GeometricContradictionEngine(
            anchors=self.layout.anchors,
            room_width=self.layout.width,
            room_height=self.layout.height,
        )

        # Multi-person instances (up to 3 simultaneous people/robots)
        self.active_targets_count: int = 1
        self.focused_target_id: str = "target_alpha"
        self.targets: Dict[str, TargetTrackInstance] = {
            p["id"]: TargetTrackInstance(p, self.layout, self.engine.anchor_map)
            for p in TARGET_PROFILES
        }

        # Backwards-compatible aliases to focused target
        self.scorer = self.targets["target_alpha"].scorer
        self.tracker = self.targets["target_alpha"].tracker
        self.ghost = self.targets["target_alpha"].ghost

        self.latest_mqtt_readings: Optional[ReadingsPayload] = None
        self.latest_mqtt_imu: Optional[IMUPayload] = None
        self.last_mqtt_rx_time: float = 0.0
        self.mqtt_connected: bool = False
        self.mqtt_client: Optional[mqtt.Client] = None

        self.last_naive_xy: Tuple[float, float] = (self.layout.width / 2.0, self.layout.height / 2.0)
        self.last_geo_xy: Tuple[float, float] = (self.layout.width / 2.0, self.layout.height / 2.0)
        self.naive_sq_errors: deque = deque(maxlen=200)
        self.geo_sq_errors: deque = deque(maxlen=200)
        self.gt_sq_errors: deque = deque(maxlen=200)

        self.latest_ws_message: Optional[WebSocketStreamMessage] = None
        self.clients: Set[WebSocket] = set()

    def start_mqtt(self) -> None:
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
        elif cmd_type == "set_target_count":
            cnt = int(cmd_data.get("count", 1))
            self.active_targets_count = max(1, min(3, cnt))
            active_ids = [p["id"] for p in TARGET_PROFILES[: self.active_targets_count]]
            if self.focused_target_id not in active_ids:
                self.focused_target_id = active_ids[0]
        elif cmd_type == "select_target":
            tid = str(cmd_data.get("target_id", "target_alpha"))
            if tid in self.targets:
                self.focused_target_id = tid
                self.scorer = self.targets[tid].scorer
                self.tracker = self.targets[tid].tracker
                self.ghost = self.targets[tid].ghost
        elif cmd_type == "reset":
            seed = int(cmd_data.get("seed", 42))
            self.sim.reset(seed)
            for inst in self.targets.values():
                inst.reset(self.layout)
            self.naive_sq_errors.clear()
            self.geo_sq_errors.clear()
            self.gt_sq_errors.clear()
        elif cmd_type == "start_replay":
            self.sim.start_scripted_chaos()
            for inst in self.targets.values():
                inst.reset(self.layout)
            self.naive_sq_errors.clear()
            self.geo_sq_errors.clear()
            self.gt_sq_errors.clear()
        elif cmd_type == "load_layout":
            new_layout = RoomLayout.model_validate(cmd_data.get("layout"))
            self.update_layout(new_layout)

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
        for inst in self.targets.values():
            inst.update_layout(new_layout, self.engine.anchor_map)

    def _step_single_target(
        self,
        inst: TargetTrackInstance,
        readings_payload: ReadingsPayload,
        imu_payload: Optional[IMUPayload],
        ground_truth_xy: Tuple[float, float],
    ) -> Tuple[dict, PeerTargetSummary]:
        ts = readings_payload.ts
        eng_out = self.engine.process(
            ts=ts,
            readings=readings_payload.readings,
            prev_pos=(inst.ghost.x, inst.ghost.y),
        )
        eng_out.scored_readings = inst.scorer.score_and_correct(eng_out.scored_readings)
        t_x, t_y, t_sigma, t_vel = inst.tracker.step(
            eng_out, imu_payload, total_anchors=len(self.layout.anchors)
        )
        recovery_advice = rank_recovery_anchors(self.layout, eng_out, (t_x, t_y))
        top_advice_id = recovery_advice[0].anchor_id if recovery_advice else None

        gx, gy, g_sigma, state, recovery_verified, conf_2m, age_ms, audit_events = inst.ghost.step(
            ts=ts,
            engine_out=eng_out,
            tracker_xy=(t_x, t_y),
            tracker_sigma=t_sigma,
            tracker_vel=t_vel,
            imu=imu_payload,
            top_advice_anchor=top_advice_id,
        )

        summary = PeerTargetSummary(
            target_id=inst.target_id,
            label=inst.label,
            color=inst.color,
            x=gx,
            y=gy,
            true_x=round(ground_truth_xy[0], 3),
            true_y=round(ground_truth_xy[1], 3),
            sigma=g_sigma,
            confidence_within_2m=conf_2m,
            state=state,
            rejected_anchors=[r.anchor_id for r in eng_out.rejected],
        )

        details = {
            "ts": ts,
            "gx": gx,
            "gy": gy,
            "g_sigma": g_sigma,
            "state": state,
            "recovery_verified": recovery_verified,
            "conf_2m": conf_2m,
            "age_ms": age_ms,
            "audit_events": audit_events,
            "eng_out": eng_out,
            "recovery_advice": recovery_advice,
            "readings_payload": readings_payload,
            "ground_truth_xy": ground_truth_xy,
        }
        return details, summary

    def tick(self) -> WebSocketStreamMessage:
        """Advance all active people/tags in the room and emit WebSocket frame."""
        now = time.time()
        self.sim._apply_scripted_chaos_step(now)

        active_profiles = TARGET_PROFILES[: self.active_targets_count]
        peer_summaries: List[PeerTargetSummary] = []
        focused_details: Optional[dict] = None

        for idx, prof in enumerate(active_profiles):
            tid = prof["id"]
            inst = self.targets[tid]

            # Target Alpha uses external MQTT telemetry if fresh, else internal simulator
            if (
                idx == 0
                and self.latest_mqtt_readings is not None
                and (now - self.last_mqtt_rx_time) < 0.45
                and not self.sim.scripted_mode
            ):
                rp = self.latest_mqtt_readings
                imu = self.latest_mqtt_imu
                gt_xy = (
                    (rp.ground_truth.x, rp.ground_truth.y)
                    if rp.ground_truth is not None
                    else (inst.trajectory.x, inst.trajectory.y)
                )
            else:
                (tx, ty), imu = inst.trajectory.step(self.sim.dt)
                imu.ts = now
                rp, _ = self.sim.compute_anchor_readings((tx, ty), now)
                rp.target_id = tid
                gt_xy = (tx, ty)

            details, summary = self._step_single_target(inst, rp, imu, gt_xy)
            peer_summaries.append(summary)
            if tid == self.focused_target_id or focused_details is None:
                focused_details = details

        assert focused_details is not None
        ts = focused_details["ts"]
        gx = focused_details["gx"]
        gy = focused_details["gy"]
        g_sigma = focused_details["g_sigma"]
        state = focused_details["state"]
        recovery_verified = focused_details["recovery_verified"]
        conf_2m = focused_details["conf_2m"]
        age_ms = focused_details["age_ms"]
        audit_events = focused_details["audit_events"]
        eng_out = focused_details["eng_out"]
        recovery_advice = focused_details["recovery_advice"]
        readings_payload = focused_details["readings_payload"]
        gt_xy = focused_details["ground_truth_xy"]

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

        tx, ty = gt_xy
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
            target_id=self.focused_target_id,
            active_targets_count=self.active_targets_count,
            peer_targets=peer_summaries,
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
        "active_targets_count": pipeline_service.active_targets_count,
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
        "active_targets_count": pipeline_service.active_targets_count,
        "focused_target_id": pipeline_service.focused_target_id,
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
