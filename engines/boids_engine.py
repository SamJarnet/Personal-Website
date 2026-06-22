import numpy as np
import random

class ContextualBandit:
    def __init__(self, env: "Boids"):
        self.env = env
        self.contexts = ["scattered", "overcrowded", "flocking", "static"]
        
        self.alpha = {c: [1, 1, 1, 1] for c in self.contexts}
        self.beta = {c: [1, 1, 1, 1] for c in self.contexts}
        self.N = {c: [0, 0, 0, 0] for c in self.contexts}
        self.Q = {c: [0.0, 0.0, 0.0, 0.0] for c in self.contexts}

    def get_context(self):
        avg_dist = self.env.find_distances()
        avg_vel = np.mean([np.linalg.norm(b["vel"]) for b in self.env.boids]) if self.env.boids else 0.0
        if avg_vel < 0.05:
            return "static"
        elif avg_dist >= 3.5:
            return "scattered"
        elif avg_dist <= 1.5:
            return "overcrowded"
        else:
            return "flocking"
            
    def get_reward(self, old_context, dist_before, dist_after):
        if old_context == "scattered":
            if dist_after < dist_before:
                return 1
        elif old_context == "overcrowded":
            if dist_after > dist_before:
                return 1
        elif old_context == "flocking":
            if 1.5 <= dist_after <= 3.5:
                return 1
        return 0

    def pull(self, action):
        # Updated from original to mutate internal parameters without relying on matplotlib sliders
        if action == 0:
            self.env.cohesion_strength = min(0.2, self.env.cohesion_strength + 0.002)
            self.env.seperation_strength = max(0.0, self.env.seperation_strength - 0.002)
        elif action == 1:
            self.env.cohesion_strength = max(0.0, self.env.cohesion_strength - 0.002)
            self.env.seperation_strength = min(0.5, self.env.seperation_strength + 0.002)
        elif action == 2:
            self.env.alignment_strength = min(0.2, self.env.alignment_strength + 0.002)
        elif action == 3:
            self.env.alignment_strength = max(0.0, self.env.alignment_strength - 0.001)

    def thomson_sample(self, context):
        P = [0.0, 0.0, 0.0, 0.0]
        for j in range(0, 4):
            P[j] = np.random.beta(self.alpha[context][j], self.beta[context][j])
        return np.argmax(P)
    
    def update(self, context, action, dist_before, dist_after):
        reward = self.get_reward(context, dist_before, dist_after)
        
        if reward == 1:
            self.alpha[context][action] += 1
        else:
            self.beta[context][action] += 1
            
        self.N[context][action] += 1
        
        current_Q = self.Q[context][action]
        self.Q[context][action] = current_Q + (reward - current_Q) / self.N[context][action]


class Boids:
    def __init__(self, cohesion=0.134, separation=0.045, alignment=0.038, boid_count=20, map_size=25.0):
        self.x_width, self.y_width = map_size, map_size
        self.boids = []
        for i in range(0, boid_count):
            self.boids.append({
                "pos": np.array([random.uniform(0, self.x_width), random.uniform(0, self.y_width)]),
                "vel": np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
            })

        self.dt = 0.1
        self.group_radius = 1.0
        self.cohesion_strength = cohesion
        self.seperation_strength = separation
        self.alignment_strength = alignment
        self.max_vel = 1.0
        
        # Instantiate contextual RL layer
        self.bandit = ContextualBandit(self)

    def check_boundaries(self):
        for i in range(0, len(self.boids)):
            pos = self.boids[i]["pos"]
            pos_x, pos_y = pos[0], pos[1]
            if pos_x <= 0: self.boids[i]["pos"][0] = self.x_width
            if pos_x >= self.x_width: self.boids[i]["pos"][0] = 0
            if pos_y <= 0: self.boids[i]["pos"][1] = self.y_width
            if pos_y >= self.y_width: self.boids[i]["pos"][1] = 0

    def find_groups(self):
        self.adjacency_list = {i: [] for i in range(len(self.boids))}
        in_group = [False] * len(self.boids)

        for i in range(0, len(self.boids)):
            for j in range(i+1, len(self.boids)):
                displacement = self.boids[i]["pos"] - self.boids[j]["pos"]
                mod_r = np.linalg.norm(displacement)

                if mod_r < self.group_radius:
                    self.adjacency_list[i].append(j)
                    self.adjacency_list[j].append(i)
                    in_group[i] = True
                    in_group[j] = True
        groups = self.create_groups(self.adjacency_list)
        return groups, in_group 

    def create_groups(self, adj_list):
        groups = []
        grouped = set()
        for i in range(len(self.boids)):
            if i in grouped:
                continue
            current_group = []
            groups.append(current_group)
            search = [i]
            while len(search) > 0:
                current_boid = search.pop(0)
                if current_boid not in grouped:
                    grouped.add(current_boid)
                    current_group.append(current_boid)
                    for j in adj_list[current_boid]:
                        if j not in grouped:
                            search.append(j)           
        return groups

    def find_average(self, groups, target):
        group_positions = []
        for group in groups: 
            if len(group) <= 1:
                group_positions.append(None)
                continue

            sum_x, sum_y = 0.0, 0.0
            group_count = len(group)
            for boid in group:
                sum_x += self.boids[boid][target][0]
                sum_y += self.boids[boid][target][1]
                
            group_positions.append(np.array([sum_x / group_count, sum_y / group_count]))
        return group_positions

    def find_distances(self):
        distances = []
        boid_count = len(self.boids)
        for i in range(0, boid_count):
            for j in range(i + 1, boid_count):
                displacement = self.boids[i]["pos"] - self.boids[j]["pos"]  
                distances.append(np.linalg.norm(displacement))
        if len(distances) == 0:
            return 0.0
        return sum(distances) / len(distances)

    def cohesion(self, groups, centers, boid):
        for i, group in enumerate(groups):
            if boid in group:
                if centers[i] is None:
                    return np.array([0.0, 0.0])
                difference = np.array(centers[i] - self.boids[boid]["pos"])
                return (difference * self.cohesion_strength)
        return np.array([0.0, 0.0])
    
    def seperation(self, groups, boid):
        for group in groups:
            if boid in group and len(group) > 1:
                vec = np.array([0.0, 0.0])
                for boid_id in group:
                    if boid_id != boid:
                        displacement = self.boids[boid]["pos"] - self.boids[boid_id]["pos"]  
                        mod_r = np.linalg.norm(displacement)
                        if 0 < mod_r < 1:
                            vec += displacement / mod_r  
                return vec * self.seperation_strength
        return np.array([0.0, 0.0])

    def allignment(self, groups, avg_vels, boid):
        for i, group in enumerate(groups):
            if boid in group:
                if avg_vels[i] is None:
                    return np.array([0.0, 0.0])
                difference = np.array(avg_vels[i] - self.boids[boid]["vel"])
                return (difference * self.alignment_strength)
        return np.array([0.0, 0.0])
    
    def cap_speed(self, i):
        boid_vel = self.boids[i]["vel"]
        mag_vel = np.linalg.norm(boid_vel)
        if mag_vel > self.max_vel:
            self.boids[i]["vel"] = (boid_vel / mag_vel) * self.max_vel

    def step(self, run_learning=False):
        self.check_boundaries()
        
        # RL Pre-step environment reading
        context, dist_before, action = None, 0.0, None
        if run_learning:
            context = self.bandit.get_context()
            dist_before = self.find_distances()
            action = self.bandit.thomson_sample(context)
            self.bandit.pull(action)

        groups, in_group = self.find_groups()
        group_avg_pos = self.find_average(groups, "pos")
        group_avg_vel = self.find_average(groups, "vel")

        for i in range(0, len(self.boids)):
            cohesion_force = self.cohesion(groups, group_avg_pos, i)
            seperation_force = self.seperation(groups, i)
            alignment_force = self.allignment(groups, group_avg_vel, i)

            self.boids[i]["vel"] += cohesion_force + seperation_force + alignment_force
            self.cap_speed(i)
            self.boids[i]["pos"] += self.boids[i]["vel"] * self.dt

        # RL Post-step environment assessment
        if run_learning and context is not None and action is not None:
            dist_after = self.find_distances()
            self.bandit.update(context, action, dist_before, dist_after)

        return group_avg_pos, in_group
    
    def run_simulation(self, total_frames=360, run_learning=False):
        history = []
        for _ in range(total_frames):
            group_averages, in_group = self.step(run_learning=run_learning)
            
            boids_data = []
            for i, boid in enumerate(self.boids):
                angle = float(np.arctan2(boid["vel"][1], boid["vel"][0]))
                boids_data.append({
                    "x": float(boid["pos"][0]),
                    "y": float(boid["pos"][1]),
                    "angle": angle,
                    "in_group": bool(in_group[i])
                })
            
            centers_data = []
            for pos in group_averages:
                if pos is not None:
                    centers_data.append({"x": float(pos[0]), "y": float(pos[1])})
                    
            history.append({
                "boids": boids_data,
                "centers": centers_data,
                # Package updated system parameters down to the web client framework
                "metrics": {
                    "cohesion": float(self.cohesion_strength),
                    "separation": float(self.seperation_strength),
                    "alignment": float(self.alignment_strength)
                }
            })
        return history