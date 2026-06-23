import numpy as np
import random

class ContextualBandit:
    def __init__(self, env: "Boids"):
        self.env = env
        self.contexts = ["scattered", "overcrowded", "flocking", "static"]

        self.action_count = 6
        self.alpha = {c: [1.0] * self.action_count for c in self.contexts}
        self.beta = {c: [1.0] * self.action_count for c in self.contexts}
        self.N = {c: [0] * self.action_count for c in self.contexts}
        self.Q = {c: [0.0] * self.action_count for c in self.contexts}

        self.decay_rate = 0.995

    def get_context(self):
        avg_dist = self.env.find_distances()
        avg_vel = np.mean([np.linalg.norm(b["vel"]) for b in self.env.boids])
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
        elif old_context == "static":
            if dist_after != dist_before:
                return 1
        return 0

    def pull(self, action):
        if action == 0:
            self.env.cohesion_strength = min(0.2, self.env.cohesion_strength + 0.002)
        elif action == 1:
            self.env.cohesion_strength = max(0.0, self.env.cohesion_strength - 0.002)
        elif action == 2:
            self.env.seperation_strength = min(0.5, self.env.seperation_strength + 0.002) # seperate into 6 actions for more specific changes
        elif action == 3:
            self.env.seperation_strength = max(0.0, self.env.seperation_strength - 0.002)
        elif action == 4:
            self.env.alignment_strength = min(0.2, self.env.alignment_strength + 0.002)
        elif action == 5:
            self.env.alignment_strength = max(0.0, self.env.alignment_strength - 0.002)

    def thomson_sample(self, context):
        P = [0.0] * self.action_count
        for j in range(0, self.action_count):
            P[j] = np.random.beta(self.alpha[context][j], self.beta[context][j])
        return( np.argmax(P))
    
    def update(self, context, action, dist_before, dist_after):
        reward = self.get_reward(context, dist_before, dist_after)
        
        if reward == 1:
            self.alpha[context][action] += 1
        else:
            self.beta[context][action] += 1
            
        self.N[context][action] += 1
        
        current_Q = self.Q[context][action]
        self.Q[context][action] = current_Q + (reward - current_Q) / self.N[context][action]

        self._decay_counts()

    # Gemini made this function, it changes the values less overtime as it finds the correct ones
    def _decay_counts(self):
        for c in self.contexts:
            for j in range(self.action_count):
                self.alpha[c][j] = 1.0 + (self.alpha[c][j] - 1.0) * self.decay_rate
                self.beta[c][j] = 1.0 + (self.beta[c][j] - 1.0) * self.decay_rate


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

        # Bandit decisions are made every bandit_K frames, not every frame —
        # a single 0.1s step is too short/noisy a horizon to attribute a
        # change in spacing to a 0.002-sized parameter nudge versus ordinary
        # boid motion or groups merging/splitting. frame_count persists on
        # self so this works correctly whether step() is driven through
        # run_simulation()'s loop or called directly per request/tick.
        self.bandit_K = 20
        self.bandit_window = []
        self.pending_action = None
        self.pending_context = None
        self.dist_before_window = None
        self.frame_count = 0

    def check_boundaries(self):
        for i in range(0, len(self.boids)):
            pos = self.boids[i]["pos"]
            pos_x, pos_y = pos[0], pos[1]
            if pos_x <= 0: self.boids[i]["pos"][0] = self.x_width
            if pos_x >= self.x_width: self.boids[i]["pos"][0] = 0
            if pos_y <= 0: self.boids[i]["pos"][1] = self.y_width
            if pos_y >= self.y_width: self.boids[i]["pos"][1] = 0

    def _pairwise_distance_matrix(self):
        """Vectorized (n,n) matrix of distances between every boid pair.
        Replaces the old nested-loop + per-pair np.linalg.norm() calls,
        which dominated frame cost (O(n^2) Python-level numpy calls)."""
        positions = np.array([b["pos"] for b in self.boids])
        diff = positions[:, None, :] - positions[None, :, :]
        return np.linalg.norm(diff, axis=-1)

    def find_groups(self):
        n = len(self.boids)
        if n == 0:
            self.adjacency_list = {}
            return [], []

        dist_matrix = self._pairwise_distance_matrix()
        np.fill_diagonal(dist_matrix, np.inf)
        adj_matrix = dist_matrix < self.group_radius

        self.adjacency_list = {i: np.nonzero(adj_matrix[i])[0].tolist() for i in range(n)}
        in_group = adj_matrix.any(axis=1).tolist()

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
        n = len(self.boids)
        if n < 2:
            return 0.0
        dist_matrix = self._pairwise_distance_matrix()
        iu = np.triu_indices(n, k=1)
        return float(dist_matrix[iu].mean())

    def cohesion(self, boid_to_group, centers, boid):
        gi = boid_to_group[boid]
        if centers[gi] is None:
            return np.array([0.0, 0.0])
        difference = np.array(centers[gi] - self.boids[boid]["pos"])
        return (difference * self.cohesion_strength)
    
    def seperation(self, boid_to_group, groups, boid):
        gi = boid_to_group[boid]
        group = groups[gi]
        if len(group) <= 1:
            return np.array([0.0, 0.0])

        vec = np.array([0.0, 0.0])
        for boid_id in group:
            if boid_id != boid:
                displacement = self.boids[boid]["pos"] - self.boids[boid_id]["pos"]  
                mod_r = np.linalg.norm(displacement)
                if 0 < mod_r < 1:
                    vec += displacement / mod_r  
        return vec * self.seperation_strength

    def allignment(self, boid_to_group, avg_vels, boid):
        gi = boid_to_group[boid]
        if avg_vels[gi] is None:
            return np.array([0.0, 0.0])
        difference = np.array(avg_vels[gi] - self.boids[boid]["vel"])
        return (difference * self.alignment_strength)
    
    def cap_speed(self, i):
        boid_vel = self.boids[i]["vel"]
        mag_vel = np.linalg.norm(boid_vel)
        if mag_vel > self.max_vel:
            self.boids[i]["vel"] = (boid_vel / mag_vel) * self.max_vel

    def step(self, run_learning=False):
        self.check_boundaries()

        # RL decision point: only fires every bandit_K frames. When a
        # window closes, score the *previous* action on the average
        # distance over the whole window it ran for, then pick + apply
        # the next action.
        if run_learning:
            if self.frame_count % self.bandit_K == 0:
                if self.pending_action is not None:
                    dist_after = (sum(self.bandit_window) / len(self.bandit_window)
                                  if self.bandit_window else self.find_distances())
                    self.bandit.update(self.pending_context, self.pending_action,
                                        self.dist_before_window, dist_after)

                self.pending_context = self.bandit.get_context()
                self.dist_before_window = self.find_distances()
                self.pending_action = self.bandit.thomson_sample(self.pending_context)
                self.bandit.pull(self.pending_action)
                self.bandit_window = []

        groups, in_group = self.find_groups()
        # O(1) lookup of which group each boid belongs to, instead of the
        # old "if boid in group" linear scan across every group for every
        # boid (an accidental O(n^2) in the force functions on top of the
        # grouping cost itself).
        boid_to_group = {b: gi for gi, group in enumerate(groups) for b in group}
        group_avg_pos = self.find_average(groups, "pos")
        group_avg_vel = self.find_average(groups, "vel")

        for i in range(0, len(self.boids)):
            cohesion_force = self.cohesion(boid_to_group, group_avg_pos, i)
            seperation_force = self.seperation(boid_to_group, groups, i)
            alignment_force = self.allignment(boid_to_group, group_avg_vel, i)

            self.boids[i]["vel"] += cohesion_force + seperation_force + alignment_force
            self.cap_speed(i)
            self.boids[i]["pos"] += self.boids[i]["vel"] * self.dt

        # RL: feed this frame's resulting distance into the open window
        if run_learning:
            self.bandit_window.append(self.find_distances())
            self.frame_count += 1

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