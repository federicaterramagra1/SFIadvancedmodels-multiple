import math
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub
from torch.ao.quantization.observer import MinMaxObserver

# ----------------------------
# Modello 
# ----------------------------
class BeanMLP(nn.Module):
    def __init__(self, in_features=16, hidden=12, num_classes=7):
        super().__init__()
        self.quant = QuantStub()
        self.fc1 = nn.Linear(in_features, hidden)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.10)  # no-op in eval; aiuta il train
        self.fc2 = nn.Linear(hidden, num_classes)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dequant(x)
        return x

    def quantize_model(self, calib_loader=None, max_calib_batches=10):
        import torch
        from torch.ao.quantization.observer import MinMaxObserver

        torch.backends.quantized.engine = 'fbgemm'
        self.to('cpu')            # <-- assicurati CPU
        self.eval()

        self.qconfig = torch.ao.quantization.QConfig(
            activation=torch.ao.quantization.default_observer,
            weight=MinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_tensor_affine)
        )

        torch.quantization.prepare(self, inplace=True)

        if calib_loader is not None:
            with torch.inference_mode():
                for i, (data, _) in enumerate(calib_loader):
                    if data.is_cuda:        # <-- evita device mismatch
                        data = data.cpu()
                    self(data)
                    if i + 1 >= max_calib_batches:
                        break

        torch.quantization.convert(self, inplace=True)
        # opzionale: flag per evitare doppie quantizzazioni
        setattr(self, "_quantized_done", True)
