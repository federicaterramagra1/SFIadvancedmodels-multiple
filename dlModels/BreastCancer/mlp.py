import torch
import torch.nn as nn
import torch.quantization as quantization

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(30, 4)  # Input features: 30, Output features: 4
        self.fc2 = nn.Linear(4, 1)   # Input features: 4, Output features: 1

        # Quantization stubs
        self.quant = quantization.QuantStub()  # Quantize input
        self.dequant = quantization.DeQuantStub()  # Dequantize output

    def forward(self, x):
        x = self.quant(x)  # Quantize input
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dequant(x)  # Dequantize output
        return x

    def quantize(self):
        """
        Apply static quantization to the model.
        """
        # Fuse layers for quantization (if applicable)
        # For SimpleMLP, there are no layers to fuse, so this step is skipped.

        # Specify quantization configuration
        self.qconfig = torch.quantization.get_default_qconfig('qnnpack')  # For CPU (use 'qnnpack' for mobile)

        # Prepare the model for static quantization
        torch.quantization.prepare(self, inplace=True)

        # Calibrate the model with dummy input
        dummy_input = torch.randn(1, 30)  # Example input with shape (batch_size, input_features)
        self(dummy_input)

        # Convert the model to a quantized version
        torch.quantization.convert(self, inplace=True)