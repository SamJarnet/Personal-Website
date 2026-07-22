import torch
import torch.nn as nn

class RocketNetwork(nn.Module):
    def __init__(self, state_size=6, action_size=6):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.network(x)

if __name__ == "__main__":
    test_model = RocketNetwork(6, 6)
    dummy_state = torch.zeros(6, dtype=torch.float32)
    with torch.no_grad():
        q_values = test_model(dummy_state)
    print("Agent network initialized successfully. Test output Q-values:", q_values)