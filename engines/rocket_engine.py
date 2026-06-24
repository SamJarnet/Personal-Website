import numpy as np

class RocketLandingEngine:
    def __init__(self, initial_pos=None, initial_vel=None):
        # Spawning at y=25.0 to give the targeting algorithm room to steer towards x=6.0
        self.rocket = {
            "pos": np.array(initial_pos if initial_pos is not None else [2.0, 25.0]), 
            "vel": np.array(initial_vel if initial_vel is not None else [0.0, 0.0])
        }  

        self.landing_pad = np.array([6.0, 1.0])
        self.dt = 0.1
        self.thrust_speed = 0.02
        self.gravity = 0.01

        self.thrusting = False
        self.landing = False
        self.landing_burn = False

    def land(self):
        height = self.rocket["pos"][1] - 1.0

        if self.rocket["vel"][1] >= 0:
            return False

        vel = abs(self.rocket["vel"][1])
        acc = (self.thrust_speed - self.gravity) * self.dt
        
        stopping_distance = (vel ** 2) / (2 * acc)
        return height <= stopping_distance

    def lookahead(self):
        future_pos = self.rocket["pos"].copy()
        future_vel = self.rocket["vel"].copy()
        for i in range(1000):
            future_vel += np.array([0.0, -self.gravity])
            future_pos += future_vel * self.dt

            if future_pos[1] < 1.0:
                future_pos[1] = 1.0
                break
        return future_pos

    def step(self):
        if self.thrusting:
            self.rocket["vel"] += np.array([0.001, self.thrust_speed])

        if self.landing and not self.landing_burn:
            self.landing_burn = self.land()

        if self.landing_burn:
            x = self.rocket["pos"][0]
            y = self.rocket["pos"][1]
            vel_y = self.rocket["vel"][1]
            
            height = y - 1.0
            acc = self.thrust_speed - self.gravity
            target_vel_y = -np.sqrt(2 * acc * height) if height > 0.0 else 0.0
            thrust_needed = (target_vel_y - vel_y) + self.gravity

            burn_time = (y - 1.0) / abs(vel_y) if abs(vel_y) > 0.001 else 1.0
            change = (self.landing_pad[0] - x) / burn_time

            x_change = (change - self.rocket["vel"][0]) * 0.01
            y_change = np.clip(thrust_needed, 0.0, self.thrust_speed)

            self.rocket["vel"] += np.array([x_change, y_change])
            angle = float(np.arctan2(y_change, x_change) - np.pi / 2)
        else:
            mod_v = np.linalg.norm(self.rocket["vel"])
            if mod_v > 0.05: 
                angle = float(np.arctan2(self.rocket["vel"][1], self.rocket["vel"][0]) - np.pi / 2)
            else:
                angle = 0.0

        # Update gravity and position
        self.rocket["vel"] += np.array([0.0, -self.gravity])
        self.rocket["pos"] += self.rocket["vel"] * self.dt

        exploded = False
        if self.rocket["pos"][1] < 1.0:
            # Check if touchdown velocity is too fast laterally or vertically
            if abs(self.rocket["vel"][1]) > 0.5 or abs(self.rocket["vel"][0]) > 0.5:
                exploded = True
            self.landing_burn, self.landing = False, False
            self.rocket["pos"][1] = 1.0
            self.rocket["vel"] = np.array([0.0, 0.0])

        return exploded, angle

    def run_simulation(self, total_frames=1, thrusting=False, landing=False):
        self.thrusting = thrusting
        self.landing = landing

        history = []
        for _ in range(total_frames):
            exploded, angle = self.step()
            
            history.append({
                "x": float(self.rocket["pos"][0]),
                "y": float(self.rocket["pos"][1]+0.5),
                "angle": angle,
                "exploded": exploded,
                "landing_pad_x": float(self.landing_pad[0]),
                "landing_pad_y": float(self.landing_pad[1]),
                "metrics": {
                    "vx": float(self.rocket["vel"][0]),
                    "vy": float(self.rocket["vel"][1]),
                    "thrusting": bool(self.thrusting),
                    "landing": bool(self.landing),
                    "landing_burn": bool(self.landing_burn)
                }
            })
        return history