"""
gridworld_engine.py
Tabular on-policy TD-control (SARSA-style update) over a fixed 5x5 maze.
Ported directly from gridworld.py — the class is untouched; only the bottom
script section is wrapped into a callable so a Flask route can drive it.
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

    def get_action(self, pos, Q, epsilon):
        if random.random() < epsilon:
            action = random.randint(0,3)
        else:
            action = np.argmax(Q[pos])
        return action


    def learn(self, eps):
        Q = np.zeros((25,4))
        alpha = 0.1
        gamma = 0.9
        epsilon = 0.05
        for i in range(0, eps):
            self.reset()
            done = False

            previous = self.fix_pos(self.pos)
            action = self.get_action(previous, Q, epsilon)

            while not done:
                pos, reward, done = self.step(action)
                next_action = self.get_action(pos, Q, epsilon)
                if done:
                    td_target = reward
                else:
                    td_target = reward + gamma * Q[pos][next_action]
                Q[previous][action] +=  alpha * (td_target - Q[previous][action])

                previous = pos
                action = next_action

            epsilon = max(0.05, epsilon * 0.9995)
        return Q


def run_simulation(eps=1000):
    """Relocated policy-extraction block from the bottom of gridworld.py."""
    test = Bandit()
    Q = test.learn(eps)
    policy = np.zeros((25,))
    for s in range(25):
        policy[s] = np.argmax(Q[s])
    policy.resize(5,5)

    return {
        "policy": policy.astype(int).tolist(),
        "grid": test.grid.tolist(),
        "q_values": Q.tolist(),
    }


if __name__ == "__main__":
    print(run_simulation()["policy"])
