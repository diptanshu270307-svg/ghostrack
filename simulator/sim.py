"""
Sensor Simulator for InSight AI / GhostTrack.
Simulates anchor ranging, multipath NLOS wall-bounce bias, and IMU telemetry.
Publishes real-time readings over MQTT and exports labeled offline datasets.
"""

import time
import math
import random
import csv
import json
import argparse
import os
from typing import List, Dict, Optional, Tuple

import paho.mqtt.client as mqtt

from shared.schemas import (
    AnchorConfig,
    WallSegment,
    RoomLayout,
    AnchorReading,
    ReadingsPayload,
    IMUPayload,
)
from simulator.geometry import check_line_of_sight
from simulator.trajectory import TrajectoryGenerator


def get_default_layout() -> RoomLayout:
    """Standard 12m x 10m demo layout with 4 anchors and 2 interior walls."""
    anchors = [
        AnchorConfig(id="A1", x=1.0, y=1.0, online=True),
        AnchorConfig(id="A2", x=11.0, y=1.0, online=True),
        AnchorConfig(id="A3", x=11.0, y=9.0, online=True),
        AnchorConfig(id="A4", x=1.0, y=9.0, online=True),
    ]
    walls = [
        # Vertical wall dividing lower part
        WallSegment(x1=5.0, y1=0.0, x2=5.0, y2=6.0),
        # Horizontal wall dividing upper corridor
        WallSegment(x1=5.0, y1=6.0, x2=9.0, y2=6.0),
    ]
    return RoomLayout(
        room_id="room_demo",
        width=12.0,
        height=10.0,
        anchors=anchors,
        walls=walls,
    )


class SensorSimulator:
    """
    Simulates RF anchors (UWB/BLE ranging) and an IMU tag.
    Computes Euclidean distance, applies NLOS wall-bounce bias and Gaussian noise,
    and publishes structured telemetry over MQTT or saves to CSV.
    """

    def __init__(
        self,
        layout: Optional[RoomLayout] = None,
        target_id: str = "target_alpha",
        noise_sigma: float = 0.15,
        nlos_bias: float = 2.50,
        rate_hz: float = 10.0,
        seed: Optional[int] = 42,
    ) -> None:
        self.layout = layout or get_default_layout()
        self.target_id = target_id
        self.noise_sigma = noise_sigma
        self.nlos_bias = nlos_bias
        self.rate_hz = rate_hz
        self.dt = 1.0 / rate_hz

        self.rng = random.Random(seed)
        self.trajectory = TrajectoryGenerator(
            room_width=self.layout.width,
            room_height=self.layout.height,
            walls=self.layout.walls,
            nominal_speed=1.2,
            seed=seed,
        )

        # Mapping of anchor_id -> online boolean
        self.anchor_status: Dict[str, bool] = {
            a.id: a.online for a in self.layout.anchors
        }

        # Path loss model parameters for RSSI (d0 = 1m, RSSI0 = -40 dBm, n = 2.2)
        self.rssi_d0 = 1.0
        self.rssi_0 = -40.0
        self.rssi_path_loss_exponent = 2.2
        self.nlos_rssi_attenuation = 14.0  # extra dBm drop through wall

    # -----------------------------------------------------------------------
    # Runtime Control Methods
    # -----------------------------------------------------------------------

    def set_noise(self, sigma: float) -> None:
        """Set measurement noise standard deviation in meters."""
        self.noise_sigma = max(0.0, float(sigma))

    def set_nlos_bias(self, bias: float) -> None:
        """Set positive distance bias added during NLOS wall occlusion."""
        self.nlos_bias = max(0.0, float(bias))

    def toggle_anchor(self, anchor_id: str, online: bool) -> bool:
        """Toggle an anchor's online status. Returns True if anchor found."""
        if anchor_id in self.anchor_status:
            self.anchor_status[anchor_id] = bool(online)
            return True
        return False

    def load_layout(self, layout: RoomLayout) -> None:
        """Apply a new room layout and refresh anchor states."""
        self.layout = layout
        self.anchor_status = {a.id: a.online for a in layout.anchors}
        self.trajectory.update_layout(layout.width, layout.height, layout.walls)

    def reset(self, seed: Optional[int] = 42) -> None:
        """Reset simulator trajectory and noise state."""
        self.rng.seed(seed)
        self.trajectory.reset(seed)
        for a in self.layout.anchors:
            self.anchor_status[a.id] = a.online

    # -----------------------------------------------------------------------
    # Sensor Synthesis
    # -----------------------------------------------------------------------

    def compute_anchor_readings(
        self, target_pos: Tuple[float, float], timestamp: float
    ) -> Tuple[ReadingsPayload, Dict[str, dict]]:
        """
        Compute distance and RSSI for all configured anchors.
        
        Returns:
            (ReadingsPayload, metadata_dict):
                metadata_dict contains ground truth distance, is_nlos, and cuts for debugging.
        """
        tx, ty = target_pos
        readings_list: List[AnchorReading] = []
        meta: Dict[str, dict] = {}

        for anchor in self.layout.anchors:
            is_online = self.anchor_status.get(anchor.id, False)

            # Ground truth Euclidean distance
            true_dist = math.hypot(tx - anchor.x, ty - anchor.y)

            # Check for wall obstructions
            has_los, cuts = check_line_of_sight((tx, ty), (anchor.x, anchor.y), self.layout.walls)

            if not is_online:
                # Offline anchor
                readings_list.append(
                    AnchorReading(
                        anchor_id=anchor.id,
                        distance=round(true_dist, 3),
                        rssi=None,
                        online=False,
                    )
                )
                meta[anchor.id] = {
                    "true_dist": true_dist,
                    "is_nlos": not has_los,
                    "cuts": cuts,
                    "online": False,
                }
                continue

            # Compute observed distance
            observed_dist = true_dist

            # Add NLOS bias if obstructed by walls (Signals lie: NLOS makes reading longer)
            if not has_los:
                observed_dist += self.nlos_bias * cuts

            # Add zero-mean Gaussian measurement noise
            noise = self.rng.gauss(0.0, self.noise_sigma)
            observed_dist += noise

            # Physical limit: distance cannot be non-positive
            observed_dist = max(0.05, observed_dist)

            # Calculate RSSI with path loss and shadowing
            d_clamped = max(0.1, true_dist)
            calc_rssi = self.rssi_0 - 10.0 * self.rssi_path_loss_exponent * math.log10(
                d_clamped / self.rssi_d0
            )
            if not has_los:
                calc_rssi -= self.nlos_rssi_attenuation * cuts
            calc_rssi += self.rng.gauss(0.0, 1.5)  # Shadowing noise

            readings_list.append(
                AnchorReading(
                    anchor_id=anchor.id,
                    distance=round(observed_dist, 3),
                    rssi=round(calc_rssi, 1),
                    online=True,
                )
            )

            meta[anchor.id] = {
                "true_dist": true_dist,
                "is_nlos": not has_los,
                "cuts": cuts,
                "online": True,
            }

        payload = ReadingsPayload(
            ts=timestamp,
            target_id=self.target_id,
            readings=readings_list,
        )
        return payload, meta

    def step(self) -> Tuple[ReadingsPayload, IMUPayload, Tuple[float, float], Dict[str, dict]]:
        """
        Advance one simulation step.
        
        Returns:
            (readings_payload, imu_payload, (true_x, true_y), anchor_meta)
        """
        now = time.time()
        (true_x, true_y), imu_payload = self.trajectory.step(self.dt)
        imu_payload.ts = now

        readings_payload, meta = self.compute_anchor_readings((true_x, true_y), now)
        return readings_payload, imu_payload, (true_x, true_y), meta

    # -----------------------------------------------------------------------
    # Offline Training Dataset Generation
    # -----------------------------------------------------------------------

    def generate_offline_dataset(
        self,
        num_samples: int,
        output_csv_path: str,
        seed: int = 100,
    ) -> None:
        """
        Generate synthetic labeled dataset for training the CNN-LSTM Tracker.
        Exports CSV containing target kinematics, anchor observations, and ground truth.
        """
        self.reset(seed)
        os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)

        fieldnames = [
            "step",
            "ts",
            "target_id",
            "true_x",
            "true_y",
            "true_vx",
            "true_vy",
            "ax",
            "ay",
            "gz",
            "anchor_id",
            "anchor_x",
            "anchor_y",
            "observed_distance",
            "true_distance",
            "rssi",
            "is_nlos",
            "online",
            "noise_sigma",
            "nlos_bias",
        ]

        with open(output_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            virtual_time = 1000000.0

            for s in range(num_samples):
                virtual_time += self.dt
                (tx, ty), imu = self.trajectory.step(self.dt)
                imu.ts = virtual_time

                readings, meta = self.compute_anchor_readings((tx, ty), virtual_time)

                for r in readings.readings:
                    anchor_cfg = next(a for a in self.layout.anchors if a.id == r.anchor_id)
                    m = meta[r.anchor_id]

                    writer.writerow({
                        "step": s,
                        "ts": virtual_time,
                        "target_id": self.target_id,
                        "true_x": round(tx, 4),
                        "true_y": round(ty, 4),
                        "true_vx": round(self.trajectory.vx, 4),
                        "true_vy": round(self.trajectory.vy, 4),
                        "ax": round(imu.ax, 4),
                        "ay": round(imu.ay, 4),
                        "gz": round(imu.gz, 4),
                        "anchor_id": r.anchor_id,
                        "anchor_x": anchor_cfg.x,
                        "anchor_y": anchor_cfg.y,
                        "observed_distance": round(r.distance, 4),
                        "true_distance": round(m["true_dist"], 4),
                        "rssi": r.rssi if r.rssi is not None else "",
                        "is_nlos": int(m["is_nlos"]),
                        "online": int(r.online),
                        "noise_sigma": self.noise_sigma,
                        "nlos_bias": self.nlos_bias,
                    })

    # -----------------------------------------------------------------------
    # Real-time MQTT Execution Loop
    # -----------------------------------------------------------------------

    def run_live(
        self,
        mqtt_host: str = "127.0.0.1",
        mqtt_port: int = 1883,
        duration: Optional[float] = None,
    ) -> None:
        """
        Connect to MQTT broker and stream readings and IMU data at rate_hz.
        """
        readings_topic = f"ghosttrack/room/{self.layout.room_id}/readings"
        imu_topic = f"ghosttrack/room/{self.layout.room_id}/imu"
        control_topic = f"ghosttrack/room/{self.layout.room_id}/control"

        client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id="ghosttrack_simulator",
        )

        def on_connect(c, userdata, flags, rc, properties=None):
            if rc == 0:
                print(f"[Simulator] Connected to MQTT broker at {mqtt_host}:{mqtt_port}")
                c.subscribe(control_topic)
                print(f"[Simulator] Subscribed to control topic: {control_topic}")
            else:
                print(f"[Simulator] Connection failed with code {rc}")

        def on_message(c, userdata, msg):
            try:
                cmd_data = json.loads(msg.payload.decode("utf-8"))
                cmd_type = cmd_data.get("command")
                if cmd_type == "set_noise":
                    self.set_noise(cmd_data.get("sigma", self.noise_sigma))
                    print(f"[Simulator] Control: set_noise -> {self.noise_sigma}")
                elif cmd_type == "set_nlos_bias":
                    self.set_nlos_bias(cmd_data.get("bias", self.nlos_bias))
                    print(f"[Simulator] Control: set_nlos_bias -> {self.nlos_bias}")
                elif cmd_type == "toggle_anchor":
                    aid = cmd_data.get("anchor_id")
                    st = cmd_data.get("online", True)
                    self.toggle_anchor(aid, st)
                    print(f"[Simulator] Control: toggle_anchor {aid} -> {st}")
                elif cmd_type == "reset":
                    seed = cmd_data.get("seed", 42)
                    self.reset(seed)
                    print(f"[Simulator] Control: reset with seed {seed}")
                elif cmd_type == "load_layout":
                    layout = RoomLayout.model_validate(cmd_data.get("layout"))
                    self.load_layout(layout)
                    print(f"[Simulator] Control: load_layout applied ({layout.width}x{layout.height})")
            except Exception as e:
                print(f"[Simulator] Error processing control message: {e}")

        client.on_connect = on_connect
        client.on_message = on_message

        print(f"[Simulator] Connecting to MQTT broker {mqtt_host}:{mqtt_port}...")
        try:
            client.connect(mqtt_host, mqtt_port, keepalive=60)
        except Exception as e:
            print(f"[Simulator] Could not connect to MQTT broker: {e}")
            raise

        client.loop_start()

        start_time = time.time()
        print(f"[Simulator] Streaming telemetry at {self.rate_hz} Hz...")

        try:
            while True:
                loop_start = time.time()
                if duration and (loop_start - start_time) > duration:
                    break

                readings_payload, imu_payload, (gx, gy), _ = self.step()

                # Publish distance readings
                client.publish(
                    readings_topic,
                    readings_payload.model_dump_json(),
                    qos=0,
                )

                # Publish IMU telemetry
                client.publish(
                    imu_topic,
                    imu_payload.model_dump_json(),
                    qos=0,
                )

                elapsed = time.time() - loop_start
                sleep_time = max(0.0, self.dt - elapsed)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n[Simulator] Stopping live simulation...")
        finally:
            client.loop_stop()
            client.disconnect()
            print("[Simulator] Disconnected.")


def main() -> None:
    parser = argparse.ArgumentParser(description="InSight AI / GhostTrack Sensor Simulator")
    parser.add_argument("--mqtt-host", type=str, default="127.0.0.1", help="MQTT Broker host")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT Broker port")
    parser.add_argument("--room-id", type=str, default="room_demo", help="Room identifier")
    parser.add_argument("--noise-sigma", type=float, default=0.15, help="Distance noise sigma in meters")
    parser.add_argument("--nlos-bias", type=float, default=2.5, help="NLOS wall bounce bias in meters")
    parser.add_argument("--rate-hz", type=float, default=10.0, help="Publish rate in Hz")
    parser.add_argument("--offline", action="store_true", help="Run in offline CSV dataset generation mode")
    parser.add_argument("--output-csv", type=str, default="data/training_data.csv", help="CSV output destination")
    parser.add_argument("--samples", type=int, default=5000, help="Number of samples for offline generation")
    parser.add_argument("--duration", type=float, default=None, help="Live simulation duration in seconds")

    args = parser.parse_args()

    layout = get_default_layout()
    layout.room_id = args.room_id

    sim = SensorSimulator(
        layout=layout,
        noise_sigma=args.noise_sigma,
        nlos_bias=args.nlos_bias,
        rate_hz=args.rate_hz,
    )

    if args.offline:
        print(f"[Simulator] Generating {args.samples} offline samples -> {args.output_csv}...")
        sim.generate_offline_dataset(args.samples, args.output_csv)
        print("[Simulator] Offline dataset generation completed successfully.")
    else:
        sim.run_live(
            mqtt_host=args.mqtt_host,
            mqtt_port=args.mqtt_port,
            duration=args.duration,
        )


if __name__ == "__main__":
    main()
