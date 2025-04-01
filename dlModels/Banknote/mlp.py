import torch.nn as nn
import torch

from torch.ao.quantization import QuantStub, DeQuantStub

class SimpleMLP(nn.Module): # 36 pesi
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.quant = QuantStub()
        self.fc1 = nn.Linear(4, 6)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(6, 2)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.dequant(x)
        return x


    def quantize_model(self, calib_loader=None):
        from torch.ao.quantization.observer import MinMaxObserver

        # Imposta la qconfig per usare per_tensor_affine per i pesi
        self.qconfig = torch.ao.quantization.QConfig(
            activation=torch.ao.quantization.default_observer,
            weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine)
        )

        # Prepara per la quantizzazione
        torch.quantization.prepare(self, inplace=True)

        # Calibrazione: passa alcuni batch per settare scale e zero_point
        self.eval()
        if calib_loader is not None:
            with torch.no_grad():
                for i, (data, _) in enumerate(calib_loader):
                    self(data)
                    if i >= 1:
                        break

        # Converte il modello quantizzato
        torch.quantization.convert(self, inplace=True)