"""
gridworld_bellman_engine.py
Model-based value iteration (Bellman optimality backup) over a fixed 5x5 maze.
Ported directly from gridworld-bellman.py — the class is untouched; only the
bottom script section is wrapped into a callable so a Flask route can drive it.
"""

import random
import numpy as np


class Bandit:
    def __init__(self):
        self.grid = np.array([['S', '.','.','.','.'], ['.', 'X','.','X','.'], ['.', '.','.','X','.'], ['.', 'X','.','.','.'], ['.', '.','.','.','G']])
        self.start = np.array([0, 0])
        self.goal = np.array([4, 4])
        self.pos = self.start.copy()
        self.actions = {
            0: np.array([0, 1]),
            1: np.array([1, 0]),
            2: np.array([0, -1]),
            3: np.array([-1, 0])
        }

    def reset(self):
        self.pos = self.start.copy()

    def fix_pos(self, pos):
        return pos[0] * 5 + pos[1]

    def step(self, action):
        reward = -1
        done = False
        move = self.actions.get(action, np.array([0 , 0]))
        future = self.pos + move

        if 0 <= future[0] < 5 and 0 <= future[1] < 5:
            future_index = self.grid[future[0], future[1]]

            if future_index != 'X':
                self.pos = future

                if future_index == 'G':
                    reward = 100
                    done = True

        return self.fix_pos(self.pos), reward, done




    def learn(self, eps):
        V = np.zeros(25)
        gamma = 0.9
        for i in range(0, eps):
            self.reset()
            done = False
            V_new = np.zeros(25)

            for s in range(25):
                row = s // 5
                col = s % 5
                if (row == 4 and col == 4) or self.grid[row, col] == 'X':
                    continue

                actions = []
                for action in [0, 1, 2, 3]:
                    self.pos = np.array([s // 5, s % 5])
                    next_pos, reward, done = self.step(action)
                    value = reward + gamma * V[next_pos]
                    actions.append(value)
                V_new[s] = max(actions)
            V = V_new.copy()
        return V


def run_simulation(eps=100):
    """Relocated reshape/print block from the bottom of gridworld-bellman.py."""
    test = Bandit()
    V = test.learn(eps)
    V.resize(5,5)

    return {
        "value_grid": V.tolist(),
        "grid": test.grid.tolist(),
    }


if __name__ == "__main__":
    print(run_simulation()["value_grid"])
