"""
Smooth 2D trajectory generator and simulated IMU sensor for InSight AI.
Generates realistic human/robot paths with smooth steering, bounded turns,
wall-bounce physics, and synthetic noisy IMU telemetry (accelerometer + gyro).
"""

from typing import Tuple, List, Optional
import math
import random
from shared.schemas import WallSegment, IMUPayload
from simulator.geometry import resolve_wall_collision


class TrajectoryGenerator:
    """
    Generates continuous smooth trajectories within room boundaries
    and synthesizes noisy IMU sensor readings (ax, ay, gz).
    """

    def __init__(
        self,
        room_width: float = 12.0,
        room_height: float = 10.0,
        walls: Optional[List[WallSegment]] = None,
        nominal_speed: float = 1.2,
        seed: Optional[int] = 42,
    ) -> None:
        self.room_width = room_width
        self.room_height = room_height
        self.walls = walls or []
        self.nominal_speed = nominal_speed
        
        self.rng = random.Random(seed)
        
        # State variables
        self.x = self.room_width * 0.25
        self.y = self.room_height * 0.25
        self.theta = self.rng.uniform(0, 2 * math.pi)
        self.speed = nominal_speed
        self.omega = 0.0  # rad/s

        # Velocities and accelerations in world coordinates
        self.vx = self.speed * math.cos(self.theta)
        self.vy = self.speed * math.sin(self.theta)
        self.ax = 0.0
        self.ay = 0.0

        # IMU biases (slow random-walk drift)
        self.bias_ax = 0.01
        self.bias_ay = -0.01
        self.bias_gz = 0.005

        # IMU noise parameters
        self.accel_noise_sigma = 0.08  # m/s^2
        self.gyro_noise_sigma = 0.015  # rad/s
        self.bias_drift_sigma = 0.0005

    def reset(self, seed: Optional[int] = None) -> None:
        """Reset trajectory state with a fresh seed or default."""
        if seed is not None:
            self.rng.seed(seed)
        self.x = self.room_width * 0.25
        self.y = self.room_height * 0.25
        self.theta = self.rng.uniform(0, 2 * math.pi)
        self.speed = self.nominal_speed
        self.omega = 0.0
        self.vx = self.speed * math.cos(self.theta)
        self.vy = self.speed * math.sin(self.theta)
        self.ax = 0.0
        self.ay = 0.0
        self.bias_ax = 0.01
        self.bias_ay = -0.01
        self.bias_gz = 0.005

    def update_layout(
        self,
        room_width: float,
        room_height: float,
        walls: List[WallSegment],
    ) -> None:
        """Update room boundary and wall constraints."""
        self.room_width = room_width
        self.room_height = room_height
        self.walls = walls
        # Ensure target remains inside new bounds
        self.x = max(0.5, min(self.room_width - 0.5, self.x))
        self.y = max(0.5, min(self.room_height - 0.5, self.y))

    def step(self, dt: float = 0.1) -> Tuple[Tuple[float, float], IMUPayload]:
        """
        Advance trajectory by dt seconds.
        
        Returns:
            ((x, y), imu_payload): Current true position and simulated noisy IMU readings.
        """
        # Smooth angular acceleration change (Ornstein-Uhlenbeck-like steering)
        omega_noise = self.rng.gauss(0.0, 0.5)
        # Pull omega back towards 0 with random steering nudges
        self.omega = 0.85 * self.omega + 0.15 * omega_noise
        self.omega = max(-1.2, min(1.2, self.omega))  # Limit max yaw rate

        prev_vx = self.vx
        prev_vy = self.vy

        # Update heading
        self.theta = (self.theta + self.omega * dt) % (2 * math.pi)

        # Slight speed variation around nominal
        speed_target = self.nominal_speed * (1.0 + 0.1 * math.sin(self.rng.uniform(0, 100)))
        self.speed = 0.9 * self.speed + 0.1 * speed_target

        self.vx = self.speed * math.cos(self.theta)
        self.vy = self.speed * math.sin(self.theta)

        # Integrate candidate position
        cand_x = self.x + self.vx * dt
        cand_y = self.y + self.vy * dt

        # Collision avoidance and bounce off walls / boundaries
        (new_x, new_y), (new_vx, new_vy) = resolve_wall_collision(
            (cand_x, cand_y),
            (self.vx, self.vy),
            self.walls,
            self.room_width,
            self.room_height,
            margin=0.4,
        )

        self.x = new_x
        self.y = new_y

        if (new_vx != self.vx) or (new_vy != self.vy):
            # Direction flipped due to collision
            self.vx = new_vx
            self.vy = new_vy
            self.theta = math.atan2(self.vy, self.vx)
            self.omega = -self.omega

        # True kinematic acceleration
        self.ax = (self.vx - prev_vx) / dt
        self.ay = (self.vy - prev_vy) / dt

        # Drift IMU biases (random walk)
        self.bias_ax += self.rng.gauss(0.0, self.bias_drift_sigma)
        self.bias_ay += self.rng.gauss(0.0, self.bias_drift_sigma)
        self.bias_gz += self.rng.gauss(0.0, self.bias_drift_sigma)

        # Synthesize noisy IMU telemetry
        noisy_ax = self.ax + self.bias_ax + self.rng.gauss(0.0, self.accel_noise_sigma)
        noisy_ay = self.ay + self.bias_ay + self.rng.gauss(0.0, self.accel_noise_sigma)
        noisy_gz = self.omega + self.bias_gz + self.rng.gauss(0.0, self.gyro_noise_sigma)

        imu = IMUPayload(
            ts=0.0,  # Caller sets appropriate timestamp
            ax=float(noisy_ax),
            ay=float(noisy_ay),
            gz=float(noisy_gz),
        )

        return (self.x, self.y), imu
