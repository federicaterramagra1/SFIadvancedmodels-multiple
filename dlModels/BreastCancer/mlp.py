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
        # Fuse layers for quantization
        self.fuse_layers()

        # Set the quantization configuration (8-bit static quantization)
        self.qconfig = quantization.default_qconfig

        # Prepare the model for static quantization
        quantization.prepare(self, inplace=True)

        # Calibrate the model (required for static quantization)
        self.calibrate()

        # Convert the model to a quantized version
        quantization.convert(self, inplace=True)

    def fuse_layers(self):
        """
        Fuse layers to prepare for quantization.
        """
        # Fuse the Linear and ReLU layers
        torch.quantization.fuse_modules(self, [['fc1', 'fc2']], inplace=True)

    def calibrate(self):
        """
        Perform calibration for static quantization.
        """
        # Use a small subset of your training data for calibration
        # Here, we use a dummy input for simplicity
        with torch.no_grad():
            dummy_input = torch.randn(1, 30)  # Adjust the input size as needed
            self(dummy_input)