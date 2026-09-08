import numpy as np

class RocketPhysics:
    def __init__(self, landing_pad=(6.0, 1.0, 0.0)):
        self.dt = 0.1
        self.thrust_speed = 0.04
        self.gravity = 0.01
        self.drag_coefficient = 0.03
        self.landing_pad = np.array(landing_pad)

        self.pos = np.array([4.0, 15.0, 0.0])
        self.vel = np.array([0.0, 0.0, 0.0])
        self.angle = np.array([0.0, 0.0, 0.0])
        self.angular_vel = np.array([0.0, 0.0, 0.0])

        self.thrusting = False
        self.landing = False
        self.landing_burn = False

        self.landed = False
        self.crashed = False

    def reset(self, pos=(4.0, 15.0, 0.0), vel=(0.0, 0.0, 0.0), angle=(0.0, 0.0, 0.0), angular_vel=(0.0, 0.0, 0.0)):
        self.pos = np.array(pos)
        self.vel = np.array(vel)
        self.angle = np.array(angle, dtype=float)
        self.angular_vel = np.array(angular_vel, dtype=float)

        self.thrusting = False
        self.landing = False
        self.landing_burn = False

    def land(self):
        height = self.pos[1] - 1.0

        if self.vel[1] >= 0:
            return False

        vel = abs(self.vel[1])
        acc = (self.thrust_speed - self.gravity)
        
        stopping_distance = (vel ** 2) / (2 * acc)
        return height <= stopping_distance

    def thrust(self, angle):
        cx, cy, cz = np.cos(angle)
        sx, sy, sz = np.sin(angle)
        
        thrust_x = self.thrust_speed * (sx * sy * cz + cx * sz)
        thrust_y = self.thrust_speed * (cx * cy)
        thrust_z = self.thrust_speed * (sx * cy)
        
        self.vel += np.array([thrust_x, thrust_y, thrust_z])


    def do_action(self, action):
        self.thrusting = action[0] > 0.05
        self.angular_vel[0] += action[1] * 0.02
        self.angular_vel[2] += action[2] * 0.02

    def step(self, action):
        self.angle += self.angular_vel
        self.angular_vel *= 0.98

        self.do_action(action)
        if self.thrusting:
            self.thrust(self.angle)

        drag = self.drag_coefficient * (self.vel ** 2) * np.sign(self.vel)

        self.vel += np.array([0.0, -self.gravity, 0.0]) - drag
        self.pos += self.vel * self.dt

        crashed = False
        landed = False

        if self.pos[1] < 1.0:
            on_pad = abs(self.pos[0] - self.landing_pad[0]) <= 0.6 and abs(self.pos[2] - self.landing_pad[2]) <= 0.6
            soft_touchdown = abs(self.vel[1]) <= 0.5 and np.linalg.norm(self.angle) <= 0.7

            if on_pad and soft_touchdown:
                landed = True
                #print("Landed")

            else:
                # print("Crashed")
                crashed = True

            self.landing_burn, self.landing = False, False
            self.pos[1] = 1.0
            self.vel = np.array([0.0, 0.0, 0.0])

        return crashed, landed