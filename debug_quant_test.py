import torch
from utils import get_network
import SETTINGS

torch.backends.quantized.engine = 'fbgemm'
device = torch.device("cpu")

# Carica e quantizza il modello come nel progetto
model = get_network(
    network_name=SETTINGS.NETWORK_NAME,
    device=device,
    dataset_name=SETTINGS.DATASET_NAME
)
model.eval()
model.to(device)

if hasattr(model, 'quantize_model'):
    model.quantize_model(calib_loader=None)
else:
    print(" Il modello non supporta la quantizzazione.")
    exit()

# Se è un QuantWrapper, estrai il modulo interno
quant_model = model.module if hasattr(model, 'module') else model

# Nome layer da testare
layer_name = "fc1"
tensor_index = (0, 0)
bits_to_flip = [2]

with torch.no_grad():
    # Accedi al layer esattamente come nel progetto
    layer = dict(quant_model.named_modules())[layer_name]
    weight, bias = layer._packed_params._weight_bias()

    scale = weight.q_scale()
    zero_point = weight.q_zero_point()

    int_value = weight[tensor_index].int_repr().item()
    float_value = weight[tensor_index].dequantize().item()

    print("[DEBUG] Valore originale:")
    print("Float:    ", float_value)
    print("Quantized:", int_value)
    print("Binario:  ", format(int_value & 0xFF, '08b'))

    # Esegui bit flip
    for bit in bits_to_flip:
        int_value ^= (1 << bit)

    int_value = max(-128, min(127, int_value))

    # Ricrea il tensore quantizzato come nel progetto
    new_q_val = torch.tensor([int_value], dtype=torch.int8)
    new_val = torch._make_per_tensor_quantized_tensor(
        new_q_val, scale=scale, zero_point=zero_point
    ).dequantize()

    # Crea nuovo tensore di pesi con valore modificato
    new_weight = weight.dequantize().clone()
    new_weight[tensor_index] = new_val

    # Riquantizza e reimposta i pesi nel layer
    q_weight = torch.quantize_per_tensor(new_weight, scale, zero_point, torch.qint8)
    layer.set_weight_bias(q_weight, bias)

    # Verifica
    new_int_value = layer._packed_params._weight_bias()[0][tensor_index].int_repr().item()
    new_float_value = layer._packed_params._weight_bias()[0][tensor_index].dequantize().item()

    print("\n[DEBUG] Valore dopo modifica:")
    print("Float:    ", new_float_value)
    print("Quantized:", new_int_value)
    print("Binario:  ", format(new_int_value & 0xFF, '08b'))
