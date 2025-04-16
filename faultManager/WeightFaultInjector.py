import struct
import torch
import SETTINGS

class WeightFaultInjector:
    def __init__(self, network):
        self.network = network.module if hasattr(network, 'module') else network
        self.golden_values = {}

    def inject_faults(self, faults, fault_mode='bit-flip'):
        for fault in faults:
            fault.layer_name = fault.layer_name.replace('module.', '')
            if (fault.layer_name, fault.tensor_index) not in self.golden_values:
                self._store_golden_value(fault.layer_name, fault.tensor_index)

            if fault_mode == 'stuck-at_params':
                self.inject_stuck_at(fault.layer_name, fault.tensor_index, fault.bits, fault.value)
            elif fault_mode == 'bit-flip':
                self.inject_bit_flip(fault.layer_name, fault.tensor_index, fault.bits)
            else:
                raise ValueError(f'Fault mode {fault_mode} non valido.')

    def _store_golden_value(self, layer_name, tensor_index):
        layer = dict(self.network.named_modules())[layer_name]
        weight, _ = layer._packed_params._weight_bias()
        self.golden_values[(layer_name, tensor_index)] = weight[tensor_index].dequantize().clone()

    def inject_bit_flip(self, layer_name: str, tensor_index: tuple, bits: list):
        self._modify_bit(layer_name, tensor_index, bits, mode="flip")

    def inject_stuck_at(self, layer_name: str, tensor_index: tuple, bits: list, value: int):
        self._modify_bit(layer_name, tensor_index, bits, mode="stuck", stuck_value=value)

    def _modify_bit(self, layer_name, tensor_index, bits, mode="flip", stuck_value=None):
        bits = [b for b in bits if 0 <= b < 8]
        if not bits:
            return

        try:
            with torch.no_grad():
                layer = dict(self.network.named_modules())[layer_name]
                weight, bias = layer._packed_params._weight_bias()

                scale = weight.q_scale()
                zero_point = weight.q_zero_point()
                int_value = weight[tensor_index].int_repr().item()

                for bit in bits:
                    if mode == "flip":
                        int_value ^= (1 << bit)
                    elif mode == "stuck":
                        if stuck_value == 1:
                            int_value |= (1 << bit)
                        else:
                            int_value &= ~(1 << bit)

                int_value = max(-128, min(127, int_value))

                print(f" BEFORE injection: {weight[tensor_index].dequantize().item()}")

                new_q_val = torch.tensor([int_value], dtype=torch.int8)
                new_val = torch._make_per_tensor_quantized_tensor(
                    new_q_val, scale=scale, zero_point=zero_point
                ).dequantize()
                new_weight = weight.dequantize().clone()
                new_weight[tensor_index] = new_val

                q_weight = torch.quantize_per_tensor(new_weight, scale, zero_point, torch.qint8)
                layer.set_weight_bias(q_weight, bias)

                print(f" AFTER injection: {layer._packed_params._weight_bias()[0][tensor_index].dequantize().item()}")

        except Exception as e:
            print(f"ERROR modifying {layer_name}, index {tensor_index}: {e}")

    def restore_golden(self):
        if not self.golden_values:
            print(" No golden values stored, skipping restore.")
            return

        for (layer_name, tensor_index), golden_value in self.golden_values.items():
            try:
                layer = dict(self.network.named_modules())[layer_name]
                weight, bias = layer._packed_params._weight_bias()

                scale = weight.q_scale()
                zero_point = weight.q_zero_point()

                restored_weight = weight.dequantize().clone()
                restored_weight[tensor_index] = golden_value.item() if isinstance(golden_value, torch.Tensor) else golden_value

                q_weight = torch.quantize_per_tensor(restored_weight, scale, zero_point, torch.qint8)
                layer.set_weight_bias(q_weight, bias)
            except Exception as e:
                print(f"ERROR restoring {layer_name}, index {tensor_index}: {e}")

        print(f" Restored {len(self.golden_values)} weights.")
        self.golden_values.clear()
