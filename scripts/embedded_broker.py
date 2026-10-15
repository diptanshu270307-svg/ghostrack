"""
Lightweight embedded asyncio MQTT 3.1.1 Broker Fallback.
Provides zero-dependency, out-of-the-box local MQTT messaging if neither
Eclipse Mosquitto nor Docker is installed on the user's system.
Supports CONNECT, CONNACK, PUBLISH (QoS 0/1), SUBSCRIBE, SUBACK, PINGREQ/PINGRESP, DISCONNECT.
"""

import asyncio
import struct
import fnmatch
from typing import Dict, Set, Tuple


def decode_remaining_length(data: bytes, offset: int = 1) -> Tuple[int, int]:
    """Decode variable length integer from MQTT header. Returns (value, bytes_consumed)."""
    multiplier = 1
    value = 0
    idx = offset
    while True:
        if idx >= len(data):
            return -1, 0
        encoded_byte = data[idx]
        value += (encoded_byte & 127) * multiplier
        idx += 1
        multiplier *= 128
        if (encoded_byte & 128) == 0:
            break
        if multiplier > 128 * 128 * 128:
            raise ValueError("Malformed remaining length in MQTT packet")
    return value, (idx - offset)


class EmbeddedMQTTBroker:
    def __init__(self, host: str = "127.0.0.1", port: int = 1883) -> None:
        self.host = host
        self.port = port
        # Map: topic_filter -> Set[asyncio.StreamWriter]
        self.subscriptions: Dict[str, Set[asyncio.StreamWriter]] = {}
        self.server: asyncio.Server = None

    def _matches(self, topic: str, topic_filter: str) -> bool:
        """Check if topic matches MQTT wildcard filter (+ and #) according to spec."""
        if topic_filter == "#" or topic == topic_filter:
            return True
        t_parts = topic.split("/")
        f_parts = topic_filter.split("/")
        for i, f in enumerate(f_parts):
            if f == "#":
                return True
            if i >= len(t_parts):
                return False
            if f != "+" and f != t_parts[i]:
                return False
        return len(t_parts) == len(f_parts)

    async def handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        client_subs: Set[str] = set()
        buffer = bytearray()

        try:
            while True:
                chunk = await reader.read(4096)
                if not chunk:
                    break
                buffer.extend(chunk)

                while len(buffer) >= 2:
                    pkt_type = (buffer[0] >> 4) & 0x0F
                    flags = buffer[0] & 0x0F
                    rem_len, header_len = decode_remaining_length(buffer, offset=1)
                    if rem_len < 0:
                        break  # Incomplete remaining length
                    total_packet_len = 1 + header_len + rem_len
                    if len(buffer) < total_packet_len:
                        break  # Incomplete packet body

                    packet = bytes(buffer[:total_packet_len])
                    del buffer[:total_packet_len]

                    payload_start = 1 + header_len

                    # Packet Types
                    if pkt_type == 1:  # CONNECT
                        # Reply CONNACK (Session Present: 0, Return Code: 0 Connection Accepted)
                        connack = b"\x20\x02\x00\x00"
                        writer.write(connack)
                        await writer.drain()

                    elif pkt_type == 3:  # PUBLISH
                        qos = (flags >> 1) & 0x03
                        curr = payload_start
                        topic_len = struct.unpack("!H", packet[curr:curr+2])[0]
                        curr += 2
                        topic = packet[curr:curr+topic_len].decode("utf-8", errors="ignore")
                        curr += topic_len

                        packet_id = None
                        if qos > 0:
                            packet_id = struct.unpack("!H", packet[curr:curr+2])[0]
                            curr += 2

                        message_bytes = packet[curr:]

                        # If QoS 1, send PUBACK
                        if qos == 1 and packet_id is not None:
                            puback = struct.pack("!BBH", 0x40, 0x02, packet_id)
                            writer.write(puback)
                            await writer.drain()

                        # Deliver to all matching subscribers
                        # Construct standard QoS 0 publish packet for subscribers
                        t_encoded = topic.encode("utf-8")
                        var_header = struct.pack("!H", len(t_encoded)) + t_encoded
                        fwd_payload = var_header + message_bytes
                        fwd_rem_len = len(fwd_payload)

                        # Encode remaining length
                        rem_bytes = bytearray()
                        x = fwd_rem_len
                        while True:
                            encoded_byte = x % 128
                            x //= 128
                            if x > 0:
                                encoded_byte |= 128
                            rem_bytes.append(encoded_byte)
                            if x == 0:
                                break

                        fwd_packet = bytes([0x30]) + bytes(rem_bytes) + fwd_payload

                        # Send to matching client writers
                        recipients = set()
                        for pattern, writers in self.subscriptions.items():
                            if self._matches(topic, pattern):
                                recipients.update(writers)

                        for rec in recipients:
                            if not rec.is_closing():
                                try:
                                    rec.write(fwd_packet)
                                except Exception:
                                    pass
                        drain_coros = [r.drain() for r in recipients if not r.is_closing()]
                        if drain_coros:
                            await asyncio.gather(*drain_coros, return_exceptions=True)

                    elif pkt_type == 8:  # SUBSCRIBE
                        curr = payload_start
                        packet_id = struct.unpack("!H", packet[curr:curr+2])[0]
                        curr += 2
                        sub_qos = []
                        while curr < len(packet):
                            t_len = struct.unpack("!H", packet[curr:curr+2])[0]
                            curr += 2
                            t_filter = packet[curr:curr+t_len].decode("utf-8", errors="ignore")
                            curr += t_len
                            req_qos = packet[curr]
                            curr += 1

                            client_subs.add(t_filter)
                            if t_filter not in self.subscriptions:
                                self.subscriptions[t_filter] = set()
                            self.subscriptions[t_filter].add(writer)
                            sub_qos.append(min(req_qos, 1))

                        # SUBACK
                        suback_payload = struct.pack("!H", packet_id) + bytes(sub_qos)
                        suback = bytes([0x90, len(suback_payload)]) + suback_payload
                        writer.write(suback)
                        await writer.drain()

                    elif pkt_type == 12:  # PINGREQ
                        # PINGRESP
                        writer.write(b"\xd0\x00")
                        await writer.drain()

                    elif pkt_type == 14:  # DISCONNECT
                        break

        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
            pass
        finally:
            for s in client_subs:
                if s in self.subscriptions and writer in self.subscriptions[s]:
                    self.subscriptions[s].discard(writer)
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    async def start(self) -> None:
        self.server = await asyncio.start_server(self.handle_client, self.host, self.port)
        addr = self.server.sockets[0].getsockname()
        print(f"[EmbeddedBroker] Fallback MQTT broker listening on {addr[0]}:{addr[1]}")
        async with self.server:
            await self.server.serve_forever()


def main() -> None:
    broker = EmbeddedMQTTBroker()
    try:
        asyncio.run(broker.start())
    except KeyboardInterrupt:
        print("\n[EmbeddedBroker] Stopped.")


if __name__ == "__main__":
    main()
