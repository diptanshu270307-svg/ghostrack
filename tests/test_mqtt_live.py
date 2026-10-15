"""
Integration test verifying live MQTT publishing and subscription between Simulator and Broker.
"""

import time
import json
import threading
import pytest
import paho.mqtt.client as mqtt

from shared.schemas import ReadingsPayload, IMUPayload
from simulator.sim import SensorSimulator, get_default_layout
from scripts.embedded_broker import EmbeddedMQTTBroker
import asyncio


def run_broker(broker: EmbeddedMQTTBroker, stop_event: threading.Event):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    server_coro = broker.start()
    task = loop.create_task(server_coro)
    
    while not stop_event.is_set():
        loop.run_until_complete(asyncio.sleep(0.05))
    
    if broker.server:
        broker.server.close()
        loop.run_until_complete(broker.server.wait_closed())
    loop.close()


def test_live_mqtt_streaming():
    test_port = 18839  # Dedicated test port
    broker = EmbeddedMQTTBroker(host="127.0.0.1", port=test_port)
    stop_event = threading.Event()
    
    broker_thread = threading.Thread(target=run_broker, args=(broker, stop_event), daemon=True)
    broker_thread.start()
    time.sleep(0.3)

    received_readings = []
    received_imu = []

    client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)

    def on_message(c, userdata, msg):
        payload = json.loads(msg.payload.decode("utf-8"))
        if "readings" in payload:
            received_readings.append(ReadingsPayload.model_validate(payload))
        elif "ax" in payload:
            received_imu.append(IMUPayload.model_validate(payload))

    client.on_message = on_message
    client.connect("127.0.0.1", test_port, keepalive=10)
    client.subscribe("ghosttrack/room/+/readings")
    client.subscribe("ghosttrack/room/+/imu")
    client.loop_start()

    # Run simulator for 1 second (10 samples)
    sim = SensorSimulator(layout=get_default_layout(), rate_hz=10.0, seed=42)
    sim_thread = threading.Thread(
        target=sim.run_live,
        kwargs={"mqtt_host": "127.0.0.1", "mqtt_port": test_port, "duration": 1.0},
        daemon=True,
    )
    sim_thread.start()
    sim_thread.join(timeout=3.0)

    time.sleep(0.3)
    client.loop_stop()
    client.disconnect()
    stop_event.set()
    broker_thread.join(timeout=1.0)

    # Verify received payloads
    assert len(received_readings) >= 5, f"Expected >= 5 readings, got {len(received_readings)}"
    assert len(received_imu) >= 5, f"Expected >= 5 imu, got {len(received_imu)}"

    first_reading = received_readings[0]
    assert len(first_reading.readings) == 4
    assert first_reading.target_id == "target_alpha"
    assert first_reading.readings[0].anchor_id == "A1"

    first_imu = received_imu[0]
    assert isinstance(first_imu.ax, float)
    assert isinstance(first_imu.ay, float)
    assert isinstance(first_imu.gz, float)
