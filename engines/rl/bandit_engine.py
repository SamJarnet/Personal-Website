"""
bandit_engine.py
Epsilon-greedy multi-armed bandit (constant step-size incremental update).
Ported directly from bandit.py — the class is untouched; only the bottom
script section is wrapped into a callable so a Flask route can drive it.
"""

import random


class Bandit:
    def __init__(self):
        self.probs = [0.1, 0.3, 0.8, 0.5]

    def pull(self, arm):
        num = random.random()
        if num < self.probs[arm]:
            return 1
        else:
            return 0

    def track(self, eps):
        total_reward = 0
        for i in range(0, eps):
            arm = random.randint(0, 3)
            reward = self.pull(arm)
            total_reward += reward
        average_reward = total_reward/eps
        return total_reward, average_reward

    def learn(self, eps):
        Q = [0.0,0.0,0.0,0.0] # estimated values of arms
        explore_threshold = 0.1
        epsilon = 0.001
        for i in range(0, eps):
            if random.random() < explore_threshold:
                C = random.randint(0, 3)
            else:
                C = Q.index(max(Q))
            R = self.pull(C)
            Q[C] = Q[C] + epsilon * (R-Q[C])
        return Q


def run_simulation(eps=100000, n=100):
    """This is exactly the Monte-Carlo averaging loop that used to sit at
    module level in bandit.py — relocated into a function so it can be
    called from a route instead of running on import."""
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
