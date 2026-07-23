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
        self.landing = False
        self.landing_burn = False

        self.landed = False
        self.crashed = False

    def reset(self, pos=(4.0, 15.0), vel=(0.0, 0.0), angle=0.0, angular_vel=0.0):
        self.pos = np.array(pos)
        self.vel = np.array(vel)
        self.angle = float(angle)
        self.angular_vel = float(angular_vel)

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

    # def lookahead(self):
    #     future_pos = self.pos.copy()
    #     future_vel = self.vel.copy()
    #     crash = False
    #     crash_vel = np.array([0.0, 0.0])
    #     crash_pos = np.array([0.0, 0.0])

    #     crash_step = 0
    #     for i in range(1000):
    #         future_vel += np.array([0.0, -self.gravity])
    #         future_pos += future_vel * self.dt

    #         if future_pos[1] < 1.0:
    #             if abs(future_vel[1]) > 0.5:
    #                 crash = True
    #                 crash_vel = future_vel
    #                 crash_step = i
    #                 crash_pos = future_pos
    #             future_pos[1] = 1.0
    #             future_vel = np.array([0.0, 0.0])

    #     return crash_pos

    def thrust(self, angle):
        thrust_x = -np.sin(angle) * self.thrust_speed
        thrust_y = np.cos(angle) * self.thrust_speed
        self.vel += np.array([thrust_x, thrust_y])



    def do_action(self, action):
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

    def step(self, action):
        self.angle += self.angular_vel
        self.angular_vel *= 0.98

        self.do_action(action)
        if self.thrusting:
            self.thrust(self.angle)

        # if self.landing:
        #     x_error = self.landing_pad[0] - self.pos[0]
        #     goal_x = x_error * 0.01 - self.vel[0] * 0.1

        #     k = 5
        #     goal_angle = -goal_x * k

        #     # limit angle 
        #     goal_angle = np.clip(goal_angle, -0.4, 0.4)

        #     angle_error = goal_angle - self.angle
        #     self.angular_vel += (0.02 * angle_error - 0.2 * self.angular_vel)

        # if self.land() and self.landing:
        #     self.thrust(self.angle)

        # update gravity air resitance and position
        drag_x = self.drag_coefficient * (self.vel[0] ** 2) * np.sign(self.vel[0])
        drag_y = self.drag_coefficient * (self.vel[1] ** 2) * np.sign(self.vel[1])

        self.vel += np.array([0.0, -self.gravity]) + np.array([-drag_x, -drag_y])
        self.pos += self.vel * self.dt

        crashed = False
        landed = False

        if self.pos[1] < 1.0:
            on_pad = abs(self.pos[0] - self.landing_pad[0]) <= 0.6
            
            soft_touchdown = abs(self.vel[1]) <= 0.35 and abs(self.angle) <= 0.35

            if on_pad and soft_touchdown:
                landed = True
                #print("Landed")

            else:
                print("Crashed")
                crashed = True

            self.landing_burn, self.landing = False, False
            self.pos[1] = 1.0
            self.vel = np.array([0.0, 0.0])

        return crashed, landed