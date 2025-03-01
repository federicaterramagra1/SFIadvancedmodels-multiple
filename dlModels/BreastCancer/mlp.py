# mlp.py
import torch
import torch.nn as nn
import torch.quantization as quantization

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(30, 4)  # Input features: 30, Output features: 4
        self.fc2 = nn.Linear(4, 1)   # Input features: 4, Output features: 1

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
