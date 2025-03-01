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

    def quantize(self):
      """
      Apply static 8-bit quantization to the model.
      """
      # Move the model to CPU (quantized models only support CPU)
      self.to('cpu')

      # Set the quantization configuration (8-bit static quantization)
      self.qconfig = torch.quantization.default_qconfig

      # Prepare the model for static quantization by adding observers
      torch.quantization.prepare(self, inplace=True)


      # Convert the model to a quantized version
      torch.quantization.convert(self, inplace=True)