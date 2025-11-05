# adapters/quant_selftest_resnet20.py
import time
import numpy as np
import torch
import SETTINGS
from utils import get_loader
from adapters.resnet20_cifar10_pretrained_int8 import get_resnet20_cifar10_int8

# -------------------- helper: float pretrained (per confronto) --------------------
def _get_resnet20_cifar10_float(device="cpu"):
    torch.hub.set_dir(torch.hub.get_dir())  # usa cache standard
    model = torch.hub.load(
        "chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True, trust_repo=True
    )
    model.eval().to(device)
    return model

# -------------------- helper: accuracy su loader completo --------------------
@torch.inference_mode()
def top1_accuracy(model, loader, device="cpu"):
    correct, total = 0, 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        pred = torch.argmax(logits, dim=1).cpu()
        correct += int((pred == yb).sum())
        total   += int(yb.size(0))
    return correct / max(1, total), correct, total

# -------------------- helper: ispezione quantizzazione --------------------
def assert_int8_quantized(model):
    """Controlli strutturali: Conv/Linear quantizzati, pesi qint8, niente observer/fakequant."""
    ok_types = (torch.nn.quantized.Conv2d, torch.nn.quantized.Linear)
    has_quant_layers = False
    for name, m in model.named_modules():
        if isinstance(m, ok_types):
            has_quant_layers = True
            try:
                w = m.weight()
                assert w.dtype == torch.qint8, f"{name}: peso NON qint8 (dtype={w.dtype})"
            except Exception as e:
                raise AssertionError(f"{name}: impossibile leggere weight() – {e}")
        # sanity: no observer/fakequant rimasti
        bad = ("Observer", "FakeQuantize", "MinMaxObserver", "MovingAverage")
        for b in bad:
            assert b not in m.__class__.__name__, f"{name}: trovato {m.__class__.__name__} (observer/fakequant)"
    assert has_quant_layers, "Nessun layer quantizzato trovato (Conv2d/Linear)."
    return True

def count_int8_weights_and_bits(model):
    tot_elems = 0
    for _, m in model.named_modules():
        if isinstance(m, (torch.nn.quantized.Conv2d, torch.nn.quantized.Linear)):
            try:
                w = m.weight()
                tot_elems += w.numel()
            except Exception:
                pass
    M = tot_elems * 8
    return tot_elems, M

# -------------------- main self-test --------------------
def run_selftest():
    torch.backends.quantized.engine = "fbgemm"
    device = torch.device("cpu")

    # Loader allineato 
    bs = getattr(SETTINGS, "BATCH_SIZE", 128)
    train_loader, _, test_loader = get_loader(dataset_name="CIFAR10", batch_size=bs, network_name="resnet20")
    print(f"[SET] test len = {len(test_loader.dataset)}  batch_size = {bs}")

    # 1) Modello FLOAT (baseline)
    fmodel = _get_resnet20_cifar10_float(device=device)
    t0 = time.time()
    acc_f, c_f, n_f = top1_accuracy(fmodel, test_loader, device=device)
    t1 = time.time()
    print(f"[FLOAT] Top-1 = {acc_f*100:.2f}%  ({c_f}/{n_f})  | eval time = {t1-t0:.2f}s")

    # 2) Modello INT8 (FX-PTQ)
    calib_batches = getattr(SETTINGS, "CALIB_BATCHES", 2)
    qmodel = get_resnet20_cifar10_int8(
        device="cpu",
        calib_loader=(train_loader if calib_batches and calib_batches > 0 else None),
        calib_batches=calib_batches,
        backend="fbgemm",
    ).to(device).eval()

    # 2.a) Check struttura quantizzata
    assert_int8_quantized(qmodel)
    n_weights, M = count_int8_weights_and_bits(qmodel)
    print(f"[COUNT] INT8 weights(no bias) = {n_weights:,}  | bit sites M = {M:,}")

    # 2.b) Accuracy INT8
    t2 = time.time()
    acc_q, c_q, n_q = top1_accuracy(qmodel, test_loader, device=device)
    t3 = time.time()
    print(f"[INT8 ] Top-1 = {acc_q*100:.2f}%  ({c_q}/{n_q})  | eval time = {t3-t2:.2f}s")

    # 3) Delta e sanity
    drop = (acc_f - acc_q) * 100
    print(f"[DELTA] Drop INT8 vs float = {drop:.2f} pp")
    print("[OK] Self-test completato senza errori strutturali.")

if __name__ == "__main__":
    run_selftest()
