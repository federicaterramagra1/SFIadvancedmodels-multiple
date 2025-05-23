import torch
import torch.nn as nn
import torch.quantization as quantization
import numpy

SEED = 42
torch.manual_seed(SEED)
numpy.random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(30, 5)
        self.fc2 = nn.Linear(5, 2)  # Cambiato da 1 a 2 classi
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
        x = self.fc2(x)  # Rimosso ReLU qui per output logit a 2 classi
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

