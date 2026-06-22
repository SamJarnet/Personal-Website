"""
bandit_ucb_engine.py
Upper-Confidence-Bound (UCB1) multi-armed bandit.
Ported directly from bandit-ucb.py — the class is untouched; only the bottom
script section is wrapped into a callable so a Flask route can drive it.
"""

import random
import numpy as np


class Bandit:
    def __init__(self):
        self.probs = [0.1, 0.3, 0.8, 0.5]

    def pull(self, arm):
        num = random.random()
        if num < self.probs[arm]:
            return 1
        else:
            return 0

    def learn(self, eps):
        Q = [0.0,0.0,0.0,0.0] # estimated values of arms
        N = [0,0,0,0] # num of pulls
        for i in range(0, 4):
                    reward = self.pull(i)
                    Q[i] += reward
                    N[i] += 1


        for t in range(0, eps):
            ucb = [0.0,0.0,0.0,0.0]
            for i in range(0, 4):
                if N[i] == 0:
                    ucb[i] = float('inf')
                else:
                    bonus = np.sqrt(2 * np.log(t + 1) / N[i])
                    ucb[i] = Q[i] / N[i] + bonus
            C = ucb.index(max(ucb))

            reward = self.pull(C)
            N[C] += 1
            Q[C] = Q[C] + (reward-Q[C]) / N[C]
        return Q


def run_simulation(eps=10000, n=100):
    """Relocated Monte-Carlo averaging loop from the bottom of bandit-ucb.py."""
    test = Bandit()
    monte_carlo = [0.0, 0.0, 0.0, 0.0]
    for i in range(0, n):
        result = test.learn(eps)
        for j in range(0, 4):
            monte_carlo[j] += result[j]/n

    return {
        "estimated_values": monte_carlo,
        "true_probs": test.probs,
    }


if __name__ == "__main__":
    print(run_simulation()["estimated_values"])
