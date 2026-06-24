import numpy as np

class RocketLandingEngine:
    def __init__(self, initial_pos=None, initial_vel=None):
        # Spawning at a higher altitude to give the guidance system room to maneuver
        self.rocket = {
            "pos": np.array(initial_pos if initial_pos is not None else [2.0, 25.0]), 
            "vel": np.array(initial_vel if initial_vel is not None else [0.0, 0.0]),
            "angle": 0.0,
            "angular_vel": 0.0
        }  

        self.landing_pad = np.array([6.0, 1.0])
        self.dt = 0.1
        self.thrust_speed = 0.02
        self.gravity = 0.01

        self.thrusting = False
        self.turn_left = False
        self.turn_right = False
        self.landing = False

    def land(self):
        height = self.rocket["pos"][1] - 1.0

        if self.rocket["vel"][1] >= 0:
            return False

        vel = abs(self.rocket["vel"][1])
        acc = (self.thrust_speed - self.gravity)
        
        stopping_distance = (vel ** 2) / (2 * acc)
        return height <= stopping_distance

    def thrust(self, angle):
        thrust_x = -np.sin(angle) * self.thrust_speed
        thrust_y = np.cos(angle) * self.thrust_speed
        self.rocket["vel"] += np.array([thrust_x, thrust_y])

    def step(self):
        # Apply manual rotational controls
        if self.turn_left:
            self.rocket["angular_vel"] += 0.01
        if self.turn_right:
            self.rocket["angular_vel"] -= 0.01

        # Update angle and apply angular friction matching rocket.py
        self.rocket["angle"] += self.rocket["angular_vel"]
        self.rocket["angular_vel"] *= 0.98

        # Manual thrusting
        if self.thrusting:
            self.thrust(self.rocket["angle"])

        # Auto-pilot lateral guidance logic
        if self.landing:
            x_error = self.landing_pad[0] - self.rocket["pos"][0]
            goal_x = x_error * 0.01 - self.rocket["vel"][0] * 0.1

            k = 5
            goal_angle = -goal_x * k

            # Limit guidance correction angle
            goal_angle = np.clip(goal_angle, -0.4, 0.4)

            angle_error = goal_angle - self.rocket["angle"]
            self.rocket["angular_vel"] += (0.02 * angle_error - 0.2 * self.rocket["angular_vel"])

        # Vertical suicide burn execution matching land() trigger criteria
        landing_burn_active = self.land()
        if landing_burn_active:
            self.thrust(self.rocket["angle"])

        # Update gravity and position
        self.rocket["vel"] += np.array([0.0, -self.gravity])
        self.rocket["pos"] += self.rocket["vel"] * self.dt

        exploded = False
        if self.rocket["pos"][1] < 1.0:
            # Crash check threshold matching rocket.py
            if abs(self.rocket["vel"][1]) > 0.5:
                exploded = True
            self.landing = False
            self.rocket["pos"][1] = 1.0
            self.rocket["vel"] = np.array([0.0, 0.0])
            self.rocket["angular_vel"] = 0.0

        return exploded, landing_burn_active

    def run_simulation(self, total_frames=1, thrusting=False, turn_left=False, turn_right=False, landing=False):
        self.thrusting = thrusting
        self.turn_left = turn_left
        self.turn_right = turn_right
        self.landing = landing

        history = []
        for _ in range(total_frames):
            exploded, landing_burn = self.step()
            
            history.append({
                "x": float(self.rocket["pos"][0]),
                "y": float(self.rocket["pos"][1]+0.5),
                "angle": -float(self.rocket["angle"]),
                "exploded": exploded,
                "landing_pad_x": float(self.landing_pad[0]),
                "landing_pad_y": float(self.landing_pad[1]),
                "metrics": {
                    "vx": float(self.rocket["vel"][0]),
                    "vy": float(self.rocket["vel"][1]),
                    "thrusting": bool(self.thrusting),
                    "landing": bool(self.landing),
                    "landing_burn": bool(landing_burn)
                }
            })
        return history