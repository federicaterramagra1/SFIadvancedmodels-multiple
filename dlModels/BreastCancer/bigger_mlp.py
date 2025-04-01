import torch
import torch.nn as nn
import torch.quantization as quantization
import numpy

SEED = 42
torch.manual_seed(SEED)
numpy.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

class BiggerMLP(nn.Module):
    def __init__(self):
        super(BiggerMLP, self).__init__()
        self.fc1 = nn.Linear(30, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 2)  # output layer
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
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
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
