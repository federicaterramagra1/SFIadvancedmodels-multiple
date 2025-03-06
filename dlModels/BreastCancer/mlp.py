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

    def forward(self, x):
        x = self.quant(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dequant(x)
        return x

    def quantize(self):
        self.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        torch.quantization.prepare(self, inplace=True)
        torch.quantization.convert(self, inplace=True)
