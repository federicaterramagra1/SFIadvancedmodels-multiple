import struct
import torch

class WeightFaultInjector:
    def __init__(self, network):
        self.network = network.module if hasattr(network, 'module') else network
        self.layer_name = None
        self.tensor_index = None
        self.bit = None
        self.golden_values = {}  # Store original weights before modification

    def inject_faults(self, faults: list, fault_mode='stuck-at'):
        for fault in faults:
            if fault.layer_name.startswith('module.'):
                fault.layer_name = fault.layer_name.replace('module.', '')
            self.inject_fault(fault, fault_mode)

    def inject_fault(self, fault, fault_mode='stuck-at'):
        self.layer_name = fault.layer_name
        self.tensor_index = fault.tensor_index
        self.bit = fault.bit
        if fault_mode == 'stuck-at':
            self.inject_stuck_at(fault.layer_name, fault.tensor_index, fault.bit, fault.value)
        elif fault_mode == 'bit-flip':
            self.inject_bit_flip(fault.layer_name, fault.tensor_index, fault.bit)
        else:
            raise ValueError(f'Invalid fault mode {fault_mode}')

    def inject_bit_flip(self, layer_name: str, tensor_index: tuple, bit: int):
        self._modify_bit(layer_name, tensor_index, bit, mode="flip")

    def inject_stuck_at(self, layer_name: str, tensor_index: tuple, bit: int, value: int):
        self._modify_bit(layer_name, tensor_index, bit, mode="stuck", stuck_value=value)

    import math

    def _modify_bit(self, layer_name: str, tensor_index: tuple, bit: int, mode="flip", stuck_value=None):
        try:
            with torch.no_grad():
                state_dict = self.network.state_dict()
                print("🔍 Available layers in state_dict():", list(state_dict.keys()))

                if f"{layer_name}._packed_params._packed_params" in state_dict:
                    packed_params = state_dict[f"{layer_name}._packed_params._packed_params"]
                    weight_tensor = packed_params[0].dequantize()

                    print(f"✅ '{layer_name}._packed_params._packed_params' shape:", weight_tensor.shape)
                    print(f"🔍 Attempting to access index: {tensor_index}")

                    if any(idx >= dim for idx, dim in zip(tensor_index, weight_tensor.shape)):
                        print(f"❌ ERROR: Tensor index {tensor_index} is out of range for layer '{layer_name}._packed_params._packed_params'.")
                        return
                else:
                    print(f"❌ ERROR: Layer '{layer_name}._packed_params._packed_params' not found in state_dict!")
                    return

                # ✅ Store golden value before modification
                if (layer_name, tensor_index) not in self.golden_values:
                    self.golden_values[(layer_name, tensor_index)] = weight_tensor[tensor_index].clone()

                print(f"✅ Accessing weight tensor at {tensor_index}: {weight_tensor[tensor_index].item()}")

                # Convert weight to integer format for bit manipulation
                weight_float = weight_tensor[tensor_index].item()
                weight_bytes = struct.pack('f', weight_float)
                weight_int = int.from_bytes(weight_bytes, byteorder='little')

                print(f"Bit {bit} before modification: {bin(weight_int)}")

                if mode == "flip":
                    weight_int ^= (1 << bit)
                elif mode == "stuck":
                    if stuck_value == 1:
                        weight_int |= (1 << bit)
                    else:
                        weight_int &= ~(1 << bit)

                print(f"Bit {bit} after modification: {bin(weight_int)}")

                # Convert modified bits back to float
                new_weight_bytes = weight_int.to_bytes(4, byteorder='little')
                new_weight_float = struct.unpack('f', new_weight_bytes)[0]

                # **Check for invalid values before assignment**
                if torch.isnan(torch.tensor(new_weight_float)) or torch.isinf(torch.tensor(new_weight_float)):
                    print(f"❌ ERROR: Modified weight at {tensor_index} resulted in an invalid value ({new_weight_float}). Skipping.")
                    return

                # **Re-quantize before re-assigning**
                scale = state_dict[f"{layer_name}.scale"]
                zero_point = state_dict[f"{layer_name}.zero_point"].item()

                quantized_weight = torch.quantize_per_tensor(torch.tensor(new_weight_float, dtype=torch.float32), scale, zero_point, torch.qint8)
                weight_tensor[tensor_index] = quantized_weight.dequantize()

                # ✅ REPACK AFTER MODIFYING WEIGHTS (Fix)
                for name, module in self.network.named_modules():
                    if isinstance(module, torch.ao.nn.quantized.Linear):
                        packed_weight = torch.ops.quantized.linear_prepack(module.weight(), module.bias())
                        module._packed_params._packed_params = packed_weight  # Correct assignment

                self.network.load_state_dict(state_dict)

                print(f"✅ Modified weight at {tensor_index}: {weight_tensor[tensor_index].item()}")

        except IndexError:
            print(f"❌ ERROR: Tensor index {tensor_index} is out of range for layer '{layer_name}._packed_params._packed_params'.")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")



    def restore_golden(self):
        if not self.golden_values:
            print('❌ CRITICAL ERROR: No golden values stored, skipping restore.')
            return

        state_dict = self.network.state_dict()

        for (layer_name, tensor_index), golden_value in self.golden_values.items():
            if f"{layer_name}._packed_params._packed_params" in state_dict:
                packed_params = state_dict[f"{layer_name}._packed_params._packed_params"]
                weight_tensor = packed_params[0].dequantize()
            elif f"{layer_name}.weight" in state_dict:
                weight_tensor = state_dict[f"{layer_name}.weight"]
            else:
                print(f"❌ ERROR: Layer '{layer_name}' not found in the network.")
                continue

            if any(index >= dim for index, dim in zip(tensor_index, weight_tensor.shape)):
                print(f"❌ ERROR: Tensor index {tensor_index} is out of range for layer '{layer_name}'.")
                continue

            # ✅ Restore the golden value and RE-QUANTIZE it
            scale = state_dict[f"{layer_name}.scale"]
            zero_point = state_dict[f"{layer_name}.zero_point"].item()

            quantized_weight = torch.quantize_per_tensor(golden_value.clone(), scale, zero_point, torch.qint8)
            weight_tensor[tensor_index] = quantized_weight.dequantize()

        # ✅ REPACK THE QUANTIZED WEIGHTS BEFORE LOADING THEM
        for name, module in self.network.named_modules():
            if isinstance(module, torch.ao.nn.quantized.Linear):
                packed_weight = torch.ops.quantized.linear_prepack(module.weight(), module.bias())
                module._packed_params._packed_params = packed_weight  # Correct assignment

        self.network.load_state_dict(state_dict)
        print(f"✅ Successfully restored golden values and repacked weights.")

        self.golden_values.clear()

