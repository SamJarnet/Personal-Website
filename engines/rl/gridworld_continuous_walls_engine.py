"""
gridworld_continuous_walls_engine.py
Value iteration over a binned continuous 5x5 space with a rectangular wall.
Ported directly from gridworld-continuous-walls.py — the class (including
plot_value_function, which is only used when this file is run directly) is
untouched; only the bottom script section is wrapped into a callable so a
Flask route can drive it without trying to pop a matplotlib window on the
server.
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Bandit:
    def __init__(self):
        self.grid_shape = np.array([5.0, 5.0]) # 5 by 5 continuous space
        self.start = np.array([0.0, 0.0])
        self.goal = np.array([4.9, 4.9])
        self.pos = self.start.copy()
        self.actions = {
            0: np.array([0, 0.1]),
            1: np.array([0.1, 0]),
            2: np.array([0, -0.1]),
            3: np.array([-0.1, 0])
        }

        self.walls = [
            np.array([2.0, 3.0, 2.0, 3.0])
        ]

    def binning(self, pos):
        bin_count = 50
        bins_x, bins_y = self.grid_shape[0]/bin_count, self.grid_shape[1]/bin_count

        return(int(pos[0]/bins_x), int(pos[1]/bins_y))

    def reset(self):
        self.pos = self.start.copy()

    def fix_pos(self, pos):
        return int(pos[0] * 50 + pos[1])

    def step(self, action):
        reward = -1
        done = False
        move = self.actions.get(action, np.array([0 , 0]))
        future = self.pos + move

        if 0.0 <= future[0] < self.grid_shape[0] and 0.0 <= future[1] < self.grid_shape[1]:

            collision = False
            for wall in self.walls:
                xmin, xmax, ymin, ymax = wall
                if xmin <= future[0] <= xmax and ymin <= future[1] <= ymax:
                    collision = True
                    break

            if not collision:
                self.pos = future
                if np.linalg.norm(self.pos - self.goal) < 0.05:
                    reward = 100
                    done = True

        binned = self.binning(self.pos)
        return self.fix_pos(binned), reward, done


    def learn(self, eps):
        V = np.zeros(2500)
        gamma = 0.9
        bin_width = 0.1

        for i in range(0, eps):
            self.reset()
            V_new = np.zeros(2500)

            for s in range(2500):
                row = s // 50
                col = s % 50
                if (row == 49 and col == 49):
                    continue

                state_pos = np.array([row * bin_width + 0.05, col * bin_width + 0.05])

                is_wall = False
                for wall in self.walls:
                    xmin, xmax, ymin, ymax = wall
                    if xmin <= state_pos[0] <= xmax and ymin <= state_pos[1] <= ymax:
                        is_wall = True
                        break
                if is_wall:
                    V_new[s] = -20.0
                    continue

                actions = []
                for action in [0, 1, 2, 3]:
                    self.pos = state_pos.copy()
                    next_pos, reward, done = self.step(action)
                    value = reward + gamma * V[next_pos]
                    actions.append(value)

                V_new[s] = max(actions)

            V = V_new.copy()
        return V

    # Gemini made this function
    def plot_value_function(self, V_grid):
        # Set up the matplotlib figure size
        plt.figure(figsize=(12, 10))

        # Create a mask to highlight the walls uniquely if you want,
        # or just let the color gradient show them naturally.
        # We use the 'viridis' color map (dark purple for low values, bright yellow for high values)
        ax = sns.heatmap(
            V_grid,
            cmap='viridis',
            cbar_kws={'label': 'State Value (V)'},
            xticklabels=10, # Show coordinate ticks every 10 bins
            yticklabels=10
        )

        plt.title('Continuous Gridworld Value Function Iteration Field', fontsize=16)
        plt.xlabel('X Bin Coordinate', fontsize=12)
        plt.ylabel('Y Bin Coordinate', fontsize=12)

        # Invert the y-axis so (0,0) starts at the bottom-left to match standard physics layouts
        plt.gca().invert_yaxis()

        # Show the plot window
        plt.show()


def run_simulation(eps=100):
    """Relocated reshape block from the bottom of gridworld-continuous-walls.py.
    Does NOT call plot_value_function — that still works if you run this file
    directly, but a web route has no display to pop a window on."""
    test = Bandit()
    V = test.learn(eps)
    V_grid = V.reshape(50, 50)
    return {
        "value_grid": V_grid.tolist(),
        "walls": [w.tolist() for w in test.walls],
        "grid_shape": test.grid_shape.tolist(),
    }


if __name__ == "__main__":
    test = Bandit()
    np.set_printoptions(threshold=np.inf, linewidth=300, precision=2, suppress=True) # type: ignore
    V = test.learn(100)
    V_grid = V.reshape(50, 50)
    test.plot_value_function(V_grid)
