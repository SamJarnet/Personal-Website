"""
bandit_contextual_engine.py
Contextual Thompson-sampling bandit. Ported directly from bandit-contextual.py
— the class is untouched; only the bottom script section is wrapped into a
callable so a Flask route can drive it.

Note: learn()'s inner loop also runs `eps` iterations (its loop variable `i`
shadows the outer one), so total pulls scale as eps**2 — that's true of the
original script as written, not something introduced here. The route layer
clamps `eps` to keep this from hanging a worker.
"""

import random
import numpy as np


class ContextualBandit:
    def __init__(self):
        self.contexts = [
            "sunny",
            "cloudy",
            "rainy"
        ]

        self.probs = {
            "sunny":  [0.1, 0.3, 0.8, 0.5],
            "cloudy": [0.7, 0.2, 0.1, 0.4],
            "rainy":  [0.2, 0.9, 0.4, 0.3]
        }

    def get_context(self):
        return random.choice(self.contexts)

    def pull(self, context, arm):
        num = random.random()
        if num < self.probs[context][arm]:
            return 1
        else:
            return 0

    def learn(self, eps):
        Q = {
            "sunny":  [0.0, 0.0, 0.0, 0.0],
            "cloudy": [0.0, 0.0, 0.0, 0.0],
            "rainy":  [0.0, 0.0, 0.0, 0.0]
        }

        N = {
            "sunny":  [0, 0, 0, 0],
            "cloudy": [0, 0, 0, 0],
            "rainy":  [0, 0, 0, 0]
        }

        alpha = {
            "sunny":  [1, 1, 1, 1],
            "cloudy": [1, 1, 1, 1],
            "rainy":  [1, 1, 1, 1]
        }

        beta = {
            "sunny":  [1, 1, 1, 1],
            "cloudy": [1, 1, 1, 1],
            "rainy":  [1, 1, 1, 1]
        }

        for i in range(eps):
            context = self.get_context()
            P = [0.0, 0.0, 0.0, 0.0]

            for i in range(0, eps):
                for j in range(0, 4):
                    P[j] = np.random.beta(alpha[context][j], beta[context][j])

                C = np.argmax(P)

                reward = self.pull(context, C)
                if reward == 1:
                    alpha[context][C] += 1
                else:
                    beta[context][C] += 1
                N[context][C] += 1

                Q[context][C] = Q[context][C] + (reward-Q[context][C]) / N[context][C]

        return Q


def run_simulation(eps=1000):
    test = ContextualBandit()
    result = test.learn(eps)
    return {
        "estimated_values": result,
        "true_probs": test.probs,
        "contexts": test.contexts,
    }


if __name__ == "__main__":
    print(run_simulation())
