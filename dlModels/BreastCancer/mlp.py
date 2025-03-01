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
        # Move the model and its parameters to CPU (quantized models only support CPU)
        self.to('cpu')

        # Ensure all parameters are on the CPU
        for param in self.parameters():
            param.data = param.data.to('cpu')

        # Set the quantization configuration (8-bit static quantization)
        self.qconfig = torch.quantization.default_qconfig

        # Prepare the model for static quantization by adding observers
        torch.quantization.prepare(self, inplace=True)

        # Calibrate the model with a representative dataset
        # Replace this with your actual calibration dataset
        with torch.no_grad():
            for _ in range(100):  # Use 100 batches for calibration
                dummy_input = torch.randn(1, 30)  # Example input (batch size: 1, input size: 30)
                self(dummy_input)

        # Convert the model to a quantized version
        torch.quantization.convert(self, inplace=True)