import numpy as np
import random


class Boids:
    def __init__(self, cohesion=0.134, separation=0.045, alignment=0.038, boid_count=20, map_size=25.0):
        self.x_width, self.y_width = map_size, map_size
        self.boids = []
        for i in range(0, boid_count):
            self.boids.append({
                "pos": np.array([random.uniform(0, self.x_width), random.uniform(0, self.y_width)]), # updated initialisation for new system
                "vel": np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
            })

        self.dt = 0.1
        self.group_radius = 1.0
        self.cohesion_strength = cohesion
        self.seperation_strength = separation
        self.alignment_strength = alignment
        self.max_vel = 1.0


    def check_boundaries(self):
        for i in range(0, len(self.boids)):
            pos = self.boids[i]["pos"]
            pos_x, pos_y = pos[0], pos[1]
            if pos_x <= 0:
                self.boids[i]["pos"][0] = self.x_width
            if pos_x >= self.x_width:
                self.boids[i]["pos"][0] = 0
            if pos_y <= 0:
                self.boids[i]["pos"][1]  = self.y_width
            if pos_y >= self.y_width:
                self.boids[i]["pos"][1] = 0

    def find_groups(self):
        self.adjacency_list = {}
        for i in range(0, len(self.boids)):
            self.adjacency_list[i] = [] 
        in_group = [False] * len(self.boids) # modified so that a bool is returned instead of changing colour

        for i in range(0, len(self.boids)):
            for j in range(i+1, len(self.boids)):
                pos_i = self.boids[i]["pos"]
                pos_j = self.boids[j]["pos"]
                displacement = pos_i - pos_j
                mod_r = np.linalg.norm(displacement)

                if mod_r < self.group_radius:  # group radius limit can be adjusted
                    self.adjacency_list[i].append(j)
                    self.adjacency_list[j].append(i)
                    in_group[i] = True
                    in_group[j] = True
        groups = self.create_groups(self.adjacency_list)
        return groups, in_group 

    def create_groups(self, adj_list):
        groups = []
        i = 0
        grouped = set()
        for i in range(len(self.boids)):
            if i in grouped:
                continue
            current_group = []
            groups.append(current_group)
            
            search = [i] # search all neighbours of each group member
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
            if len(group) <= 1: # filter out boids not in groups
                group_positions.append(None)
                continue

            sum_x, sum_y = 0.0, 0.0
            group_count = len(group)

            if len(group) > 1:
                for boid in group:
                    sum_x += self.boids[boid][target][0]
                    sum_y += self.boids[boid][target][1]
                    
                avg_x = sum_x / group_count
                avg_y = sum_y / group_count
                group_positions.append(np.array([avg_x, avg_y]))

        return group_positions
    
    

    def cohesion(self, groups, centers, boid):
        for i, group in enumerate(groups):
            if boid in group:
                if centers[i] is None:
                    return np.array([0.0, 0.0])
                
                difference = np.array(centers[i] - self.boids[boid]["pos"]) # get vector in direction of average point
                return (difference * self.cohesion_strength)
            
        return np.array([0.0, 0.0])
    
    def seperation(self, groups, boid):
        for group in groups:
            if boid in group and len(group) > 1:
                vec = np.array([0.0, 0.0])

                for boid_id in group:
                    if boid_id != boid:
                        displacement = self.boids[boid]["pos"] - self.boids[boid_id]["pos"]  
                        mod_r = np.linalg.norm(displacement)  # direction to each other boid

                        if 0 < mod_r < 1:
                            vec += displacement / mod_r  

                return vec * self.seperation_strength
        return np.array([0.0, 0.0])
    

    def allignment(self, groups, avg_vels, boid):
        for i, group in enumerate(groups):
            if boid in group:
                if avg_vels[i] is None:
                    return np.array([0.0, 0.0])
                
                difference = np.array(avg_vels[i] - self.boids[boid]["vel"]) # get vector of average vels
                return (difference * self.alignment_strength)
            
        return np.array([0.0, 0.0])
    
    def cap_speed(self, i):
        boid_vel = self.boids[i]["vel"]
        mag_vel = np.linalg.norm(boid_vel)
        if  mag_vel > self.max_vel:
            self.boids[i]["vel"] = (boid_vel/mag_vel) * self.max_vel

    def step(self):
        self.check_boundaries()
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

        return group_avg_pos, in_group
    

    def run_simulation(self, total_frames=360):
        """Generated by Gemini to map my project to JSON"""
        history = []
        for _ in range(total_frames):
            group_averages, in_group = self.step()
            
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
                "centers": centers_data
            })
        return history