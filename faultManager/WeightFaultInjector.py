import struct
import torch

class WeightFaultInjector:
    def __init__(self, network):
        self.network = network.module if hasattr(network, 'module') else network
        self.layer_name = None
        self.tensor_index = None
        self.bits = None  # Update to handle multiple bits
        self.golden_values = {}  # Store original weights before modification
        self.has_restored = False  # Track if restore_golden() was already executed

    def inject_faults(self, faults: list, fault_mode='stuck-at'):
        for fault in faults:
            if fault.layer_name.startswith('module.'):
                fault.layer_name = fault.layer_name.replace('module.', '')

            if fault_mode == 'stuck-at':
                self.inject_stuck_at(fault.layer_name, fault.tensor_index, fault.bits, fault.value)
            elif fault_mode == 'bit-flip':
                self.inject_bit_flip(fault.layer_name, fault.tensor_index, fault.bits)
            else:
                raise ValueError(f'Invalid fault mode {fault_mode}')


    def inject_fault(self, fault, fault_mode='stuck-at'):
        self.layer_name = fault.layer_name
        self.tensor_index = fault.tensor_index
        self.bits = fault.bits  # Handle bits as a list
        if fault_mode == 'stuck-at':
            self.inject_stuck_at(fault.layer_name, fault.tensor_index, fault.bits, fault.value)
        elif fault_mode == 'bit-flip':
            self.inject_bit_flip(fault.layer_name, fault.tensor_index, fault.bits)
        else:
            raise ValueError(f'Invalid fault mode {fault_mode}')

    def inject_bit_flip(self, layer_name: str, tensor_index: tuple, bits: list):
        self._modify_bit(layer_name, tensor_index, bits, mode="flip")

    def inject_stuck_at(self, layer_name: str, tensor_index: tuple, bits: list, value: int):
        self._modify_bit(layer_name, tensor_index, bits, mode="stuck", stuck_value=value)

    def _modify_bit(self, layer_name: str, tensor_index: tuple, bits: list, mode="flip", stuck_value=None):
        try:
            with torch.no_grad():
                state_dict = self.network.state_dict()
                weight_tensor = state_dict[f"{layer_name}._packed_params._packed_params"][0].dequantize()

                weight_float = weight_tensor[tensor_index].item()
                weight_bytes = struct.pack('f', weight_float)
                weight_int = int.from_bytes(weight_bytes, byteorder='little')

                # Modifica bit multipli chiaramente:
                for bit in bits:
                    if mode == "flip":
                        weight_int ^= (1 << bit)  # Flip del bit
                    elif mode == "stuck":
                        if stuck_value == 1:
                            weight_int |= (1 << bit)  # set a 1
                        else:
                            weight_int &= ~(1 << bit)  # set a 0

                new_weight_bytes = weight_int.to_bytes(4, byteorder='little')
                new_weight_float = struct.unpack('f', new_weight_bytes)[0]

                scale = state_dict[f"{layer_name}.scale"]
                zero_point = state_dict[f"{layer_name}.zero_point"].item()
                quantized_weight = torch.quantize_per_tensor(
                    torch.tensor(new_weight_float, dtype=torch.float32), scale, zero_point, torch.qint8
                )

                # Applica chiaramente il peso modificato:
                weight_tensor[tensor_index] = quantized_weight.dequantize()

                # Ricompatta i parametri quantizzati chiaramente:
                for name, module in self.network.named_modules():
                    if isinstance(module, torch.ao.nn.quantized.Linear):
                        module._packed_params._packed_params = torch.ops.quantized.linear_prepack(module.weight(), module.bias())

                self.network.load_state_dict(state_dict)

        except Exception as e:
            print(f"Errore inatteso in _modify_bit: {e}")



    def restore_golden(self):
        if not self.golden_values:
            if not self.has_restored:
                print(" CRITICAL ERROR: No golden values stored, skipping restore.")
                self.has_restored = True
            return

        state_dict = self.network.state_dict()
        total_restored = 0

        print(f" Attempting to restore {len(self.golden_values)} golden values...")

        for (layer_name, tensor_index), golden_value in list(self.golden_values.items()):
            if f"{layer_name}._packed_params._packed_params" in state_dict:
                packed_params = state_dict[f"{layer_name}._packed_params._packed_params"]
                weight_tensor = packed_params[0].dequantize()
            elif f"{layer_name}.weight" in state_dict:
                weight_tensor = state_dict[f"{layer_name}.weight"]
            else:
                print(f"ERROR: Layer '{layer_name}' not found in state_dict.")
                continue

            scale = state_dict[f"{layer_name}.scale"]
            zero_point = state_dict[f"{layer_name}.zero_point"].item()

            quantized_weight = torch.quantize_per_tensor(golden_value.clone(), scale, zero_point, torch.qint8)
            weight_tensor[tensor_index] = quantized_weight.dequantize()
            total_restored += 1

        # Repack the quantized weights
        for name, module in self.network.named_modules():
            if isinstance(module, torch.ao.nn.quantized.Linear):
                module._packed_params._packed_params = torch.ops.quantized.linear_prepack(module.weight(), module.bias())

        self.network.load_state_dict(state_dict)
        print(f" Successfully restored {total_restored} golden values.")

        # Clear the golden values after restore to avoid repeated restore attempts
        self.golden_values.clear()

