import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub

class LetterMLP(nn.Module):  
    def __init__(self):
        super(LetterMLP, self).__init__()
        self.quant = QuantStub()
        
        self.fc1 = nn.Linear(16, 32)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(16, 7)
        
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        x = self.dequant(x)
        return x

    def quantize_model(self, calib_loader=None):
        from torch.ao.quantization.observer import MinMaxObserver

        self.qconfig = torch.ao.quantization.QConfig(
            activation=torch.ao.quantization.default_observer,
            weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine)
        )

        torch.quantization.prepare(self, inplace=True)

        self.eval()
        if calib_loader is not None:
            with torch.no_grad():
                for i, (data, _) in enumerate(calib_loader):
                    self(data)
                    if i >= 1:
                        break

        torch.quantization.convert(self, inplace=True)
