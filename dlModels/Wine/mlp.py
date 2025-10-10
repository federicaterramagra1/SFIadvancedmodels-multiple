import torch
import torch.nn as nn
import torch.nn.quantized as nnq
import torch.nn.intrinsic.quantized as nniq
from torch.ao.quantization import QuantStub, DeQuantStub
from torch.ao.quantization import get_default_qconfig, prepare, convert, fuse_modules
from torch.ao.quantization.observer import MovingAverageMinMaxObserver, PerChannelMinMaxObserver
from torch.ao.quantization.qconfig import QConfig

DEBUG_Q = True  # metti False per silenziare

class WineMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.fc1 = nn.Linear(13, 6)   # 13×6 = 78 pesi
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(6, 3)    # 6×3 = 18 pesi
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.relu1(self.fc1(x))
        x = self.fc2(x)
        x = self.dequant(x)
        return x

    @torch.inference_mode()
    def quantize_model(self, calib_loader=None, max_calib_batches=None):
        torch.backends.quantized.engine = "fbgemm"
        self.eval(); self.cpu()

        # 1) Fuse (Linear+ReLU) -> migliora quant e latenza
        try:
            fuse_modules(self, [["fc1", "relu1"]], inplace=True)
        except Exception as e:
            if DEBUG_Q: print(f"[PTQ] fuse_modules skipped: {e}")

        # 2) qconfig robusto (attn per-tensor, pesi per-channel)
        self.qconfig = get_default_qconfig("fbgemm")

        # 3) prepare + calibrazione su dati rappresentativi (float32 su CPU)
        prepare(self, inplace=True)

        if calib_loader is not None:
            seen = 0
            for xb, _ in calib_loader:
                self(xb.to("cpu").float())
                seen += 1
                if max_calib_batches is not None and seen >= max_calib_batches:
                    break

        # 4) convert
        convert(self, inplace=True)
        setattr(self, "_quantized_done", True)

        if DEBUG_Q:
            try:
                qmods = []
                for n, m in self.named_modules():
                    if isinstance(m, (nnq.Linear, nniq.LinearReLU)):
                        try:
                            sch = m.weight().qscheme()
                        except Exception:
                            sch = "n/a"
                        qmods.append((n, type(m).__name__, sch))
                print("[PTQ] Quantized linears:", qmods)
            except Exception as e:
                print(f"[PTQ][DEBUG] ispezione fallita: {e}")

