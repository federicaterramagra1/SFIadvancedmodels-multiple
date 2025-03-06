import torch
import torch.nn as nn
import torch.quantization as quantization

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(30, 4)
        self.fc2 = nn.Linear(4, 1)
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.quant(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dequant(x)
        return x

    def quantize_model(self):
        self.qconfig = torch.quantization.get_default_qconfig('qnnpack')
        torch.quantization.prepare(self, inplace=True)
        # Calibration step here if needed
        torch.quantization.convert(self, inplace=True)