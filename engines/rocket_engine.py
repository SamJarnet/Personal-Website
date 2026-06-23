import numpy as np

class RocketLandingEngine:
    def __init__(self, initial_pos=None, initial_vel=None):
        self.rocket = {
            "pos": np.array(initial_pos if initial_pos is not None else [2.0, 1.0]), 
            "vel": np.array(initial_vel if initial_vel is not None else [0.0, 0.0])
        }  

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
        acc = self.thrust_speed - self.gravity
        
        # Suicide burn calculation
        stopping_distance = (vel ** 2) / (2 * acc)

        return height <= stopping_distance

    def lookahead(self):
        future_pos = self.rocket["pos"].copy()
        future_vel = self.rocket["vel"].copy()
        crash = False
        crash_vel = np.array([0.0, 0.0])
        crash_step = 0
        for i in range(1000):
            future_vel += np.array([0.0, -self.gravity])
            future_pos += future_vel * self.dt

            if future_pos[1] < 1.0:
                if abs(future_vel[1]) > 0.5:
                    crash = True
                    crash_vel = future_vel
                    crash_step = i
                future_pos[1] = 1.0
                future_vel = np.array([0.0, 0.0])
        return crash, crash_vel, crash_step

    def step(self):
        if self.thrusting:
            self.rocket["vel"] += np.array([0.001, self.thrust_speed])

        if self.landing:
            self.landing_burn = self.land()
        
        if self.landing_burn:
            self.rocket["vel"] += np.array([0.0, self.thrust_speed])

        self.rocket["vel"] += np.array([0.0, -0.01])
        self.rocket["pos"] += self.rocket["vel"] * self.dt

        exploded = False
        if self.rocket["pos"][1] < 1.0:
            if abs(self.rocket["vel"][1]) > 0.5:
                exploded = True
            self.landing_burn, self.landing = False, False
            self.rocket["pos"][1] = 1.0
            self.rocket["vel"] = np.array([0.0, 0.0])
            
        return exploded

    def run_simulation(self, total_frames=360, thrusting=False, landing=False):
        """
        Generates web-ready JSON-serializable telemetry arrays 
        matching the operational model of the Boids framework.
        """
        self.thrusting = thrusting
        self.landing = landing

        history = []
        for _ in range(total_frames):
            exploded = self.step()
        
            
            history.append({
                "x": float(self.rocket["pos"][0]),
                "y": float(self.rocket["pos"][1]+0.15),
                "angle": np.pi/2,
                "exploded": exploded,
                "metrics": {
                    "vx": float(self.rocket["vel"][0]),
                    "vy": float(self.rocket["vel"][1]),
                    "thrusting": bool(self.thrusting),
                    "landing": bool(self.landing),
                    "landing_burn": bool(self.landing_burn)
                }
            })
        return history