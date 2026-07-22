import numpy as np

class RocketPhysics:
    def __init__(self, landing_pad=(6.0, 1.0)):
        self.dt = 0.1
        self.thrust_speed = 0.04
        self.gravity = 0.01
        self.drag_coefficient = 0.03
        self.landing_pad = np.array(landing_pad)

        self.pos = np.array([4.0, 15.0])
        self.vel = np.array([0.0, 0.0])
        self.angle = 0.0
        self.angular_vel = 0.0

        self.thrusting = False
        self.landed = False
        self.crashed = False

    def reset(self, pos=(4.0, 15.0), vel=(0.0, 0.0), angle=0.0, angular_vel=0.0):
        self.pos = np.array(pos, dtype=np.float64)
        self.vel = np.array(vel, dtype=np.float64)
        self.angle = float(angle)
        self.angular_vel = float(angular_vel)
        self.thrusting = False
        self.landed = False
        self.crashed = False

    def land(self):
        """Calculates suicide burn timing based on altitude and descent velocity."""
        height = self.pos[1] - 1.0
        if self.vel[1] >= 0:
            return False

        vel = abs(self.vel[1])
        acc = (self.thrust_speed - self.gravity)
        stopping_distance = (vel ** 2) / (2 * acc)
        return height <= stopping_distance

    def thrust(self, angle):
        thrust_x = -np.sin(angle) * self.thrust_speed
        thrust_y = np.cos(angle) * self.thrust_speed
        self.vel += np.array([thrust_x, thrust_y])

    def do_action(self, action):
        """
        Executes discrete actions (0-5) for DQN or Human Manual input:
        0: Idle
        1: Main Thrust
        2: Rotate Left
        3: Rotate Right
        4: Main Thrust + Rotate Left
        5: Main Thrust + Rotate Right
        """
        self.thrusting = False

        if action == 1:
            self.thrusting = True
        elif action == 2:
            self.angular_vel += 0.01
        elif action == 3:
            self.angular_vel -= 0.01
        elif action == 4:
            self.thrusting = True
            self.angular_vel += 0.01
        elif action == 5:
            self.thrusting = True
            self.angular_vel -= 0.01

    def _update_physics_and_collisions(self):
        """Applies drag, gravity, position update, and touchdown/crash logic."""
        drag_x = self.drag_coefficient * (self.vel[0] ** 2) * np.sign(self.vel[0])
        drag_y = self.drag_coefficient * (self.vel[1] ** 2) * np.sign(self.vel[1])

        self.vel += np.array([0.0, -self.gravity]) + np.array([-drag_x, -drag_y])
        self.pos += self.vel * self.dt

        crashed = False
        landed = False

        if self.pos[1] < 1.0:
            on_pad = abs(self.pos[0] - self.landing_pad[0]) <= 1.0
            soft_touchdown = abs(self.vel[1]) <= 0.3 and abs(self.angle) <= 0.3

            if on_pad and soft_touchdown:
                landed = True
                self.landed = True
            else:
                crashed = True
                self.crashed = True

            self.pos[1] = 1.0
            self.vel = np.array([0.0, 0.0])

        return crashed, landed

    def step(self, action):
        """Step function for PyTorch DQN AI or Manual Human controls."""
        self.angle += self.angular_vel
        self.angular_vel *= 0.98

        self.do_action(action)
        if self.thrusting:
            self.thrust(self.angle)

        return self._update_physics_and_collisions()

    def step_traditional(self):
        """
        TRADITIONAL ALGORITHM: Uses PD-like lateral guidance 
        and suicide burn logic without neural networks.
        """
        x_error = self.landing_pad[0] - self.pos[0]
        goal_x = x_error * 0.01 - self.vel[0] * 0.1

        k = 5
        goal_angle = -goal_x * k
        goal_angle = np.clip(goal_angle, -0.4, 0.4)

        angle_error = goal_angle - self.angle
        self.angular_vel += (0.02 * angle_error - 0.2 * self.angular_vel)

        self.angle += self.angular_vel
        self.angular_vel *= 0.98

        self.thrusting = False
        if self.land():
            self.thrust(self.angle)
            self.thrusting = True

        return self._update_physics_and_collisions()