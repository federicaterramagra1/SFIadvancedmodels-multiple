# adapters/builders.py
import numpy as np
import torch
import SETTINGS
from utils import get_loader
from adapters.resnet20_cifar10_pretrained_int8 import get_resnet20_cifar10_int8

def build_resnet20_cifar10_pretrained_int8_and_clean():
    """ResNet20@CIFAR10 INT8 pretrained (CPU, fbgemm) + clean pass, baseline & accuracy."""
    torch.backends.quantized.engine = "fbgemm"
    device = torch.device("cpu")

    bs = getattr(SETTINGS, "BATCH_SIZE", 128)
    calib_batches = getattr(SETTINGS, "CALIB_BATCHES", 2)

    # Loader (test con shuffle=False per allineamento clean/faulty)
    train_loader, _, test_loader = get_loader(
        dataset_name="CIFAR10",
        batch_size=bs,
        network_name="resnet20",
    )

    # Quantizzazione FX-PTQ
    model = get_resnet20_cifar10_int8(
        device="cpu",
        calib_loader=(train_loader if calib_batches and calib_batches > 0 else None),
        calib_batches=calib_batches,
        backend="fbgemm",
    )
    model.eval().to(device)

    # Clean pass + accuracy
    clean_by_batch = []
    correct, total = 0, 0
    with torch.inference_mode():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1).cpu()
            clean_by_batch.append(pred)

            # accuracy quantizzata
            if isinstance(yb, torch.Tensor):
                correct += (pred == yb).sum().item()
                total   += yb.size(0)

    quant_top1 = correct / max(1, total)
    print(f"[QUANT] ResNet20 INT8 – Top-1 test accuracy: {quant_top1*100:.2f}%  ({correct}/{total})")
    print(f"[CHECK] test samples seen = {total} (dataset len = {len(test_loader.dataset)})")

    # Baseline distribution (serve per la FI)
    clean_flat = (torch.cat(clean_by_batch, dim=0)
                  if len(clean_by_batch) else torch.tensor([], dtype=torch.long))
    num_classes = int(clean_flat.max().item()) + 1 if clean_flat.numel() > 0 else 10
    baseline_hist = np.bincount(clean_flat.numpy(), minlength=num_classes) if clean_flat.numel() > 0 else np.zeros(num_classes, dtype=int)
    baseline_dist = baseline_hist / max(1, baseline_hist.sum())
    print(f"[BASELINE] pred dist = {baseline_dist.tolist()}")

    return model, device, test_loader, clean_by_batch, baseline_hist, baseline_dist, num_classes
