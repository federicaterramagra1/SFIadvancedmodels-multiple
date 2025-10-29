# adapters/resnet20_cifar10_pretrained_int8.py

import torch
from torch.ao.quantization import get_default_qconfig
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx

def _load_resnet20_cifar10_pretrained(device="cpu"):
    # Modello float pretrained (repo chenyaofo)
    model = torch.hub.load(
        'chenyaofo/pytorch-cifar-models',
        'cifar10_resnet20',
        pretrained=True,
        trust_repo=True  # evita prompt interattivo
    )
    model.eval().to("cpu")  # PTQ: tutto su CPU (fbgemm)
    return model

@torch.no_grad()
def get_resnet20_cifar10_int8(device="cpu", calib_loader=None, calib_batches=2, backend="fbgemm"):
    """
    ResNet20 CIFAR10 quantizzata INT8 via FX-PTQ (CPU).
    FX gestisce automaticamente l'add residuo (dequant/quant intorno) → niente errori QuantizedCPU.
    """
    torch.backends.quantized.engine = backend
    device = torch.device("cpu")

    # 1) modello float
    model = _load_resnet20_cifar10_pretrained(device=device)

    # 2) qconfig + prepare FX
    qconfig = get_default_qconfig(backend)
    qconfig_dict = {"": qconfig}
    example_inputs = (torch.randn(1, 3, 32, 32),)
    prepared = prepare_fx(model, qconfig_dict, example_inputs=example_inputs)
    prepared.eval()

    # 3) calibrazione leggera
    if calib_loader is not None and calib_batches and calib_batches > 0:
        for i, (xb, _) in enumerate(calib_loader):
            prepared(xb.to(device, non_blocking=True))
            if i + 1 >= calib_batches:
                break

    # 4) convert FX → INT8
    quantized = convert_fx(prepared)
    quantized.eval()
    return quantized  # CPU

__all__ = ["get_resnet20_cifar10_int8"]
