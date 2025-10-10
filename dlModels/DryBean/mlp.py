import math
import numpy as np
from collections import Counter
import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub
from torch.ao.quantization import get_default_qconfig, prepare, convert  

# ----------------------------
# Modello 
# ----------------------------
class BeanMLP(nn.Module):
    def __init__(self, in_features=16, hidden=12, num_classes=7):
        super().__init__()
        self.quant = QuantStub()
        self.fc1 = nn.Linear(in_features, hidden)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=0.10)  # no-op in eval
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

    @torch.inference_mode()
    def quantize_model(self, calib_loader=None, max_calib_batches=None):
        torch.backends.quantized.engine = "fbgemm"
        self.eval(); self.cpu()

        # default: act per-tensor, weight per-channel
        self.qconfig = get_default_qconfig("fbgemm")
        prepare(self, inplace=True)

        if calib_loader is not None:
            for i, (xb, _) in enumerate(calib_loader):
                self(xb.to("cpu").float())
                if max_calib_batches is not None and (i + 1) >= max_calib_batches:
                    break

        convert(self, inplace=True)
        setattr(self, "_quantized_done", True)

        # debug opzionale
        import torch.nn.quantized as nnq
        for n, m in self.named_modules():
            if isinstance(m, nnq.Linear):
                print(f"[PTQ] {n}: qscheme={m.weight().qscheme()}")
