import struct
import torch

class WeightFaultInjector:
    def __init__(self, network):
        self.network = network.module if hasattr(network, 'module') else network
        self.layer_name = None
        self.tensor_index = None
        self.bit = None
        self.golden_values = {}  # Initialize golden values dictionary

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

    def _modify_bit(self, layer_name: str, tensor_index: tuple, bit: int, mode="flip", stuck_value=None):
        try:
            with torch.no_grad():
                state_dict = self.network.state_dict()

                # ✅ Debugging: Print available layers
                print("Available layers in state_dict():", state_dict.keys())

                # ✅ Handle quantized layers
                if f"module.{layer_name}._packed_params._packed_params" in state_dict:
                    layer_name = f"module.{layer_name}._packed_params._packed_params"
                elif f"{layer_name}._packed_params._packed_params" in state_dict:
                    layer_name = f"{layer_name}._packed_params._packed_params"
                else:
                    print(f"ERROR: Layer '{layer_name}' not found in the network.")
                    return

                # ✅ Extract weight tensor from packed parameters
                packed_params = state_dict[layer_name]
                weight_tensor = packed_params[0].dequantize()  # Convert to floating point

                # ✅ Check if tensor is 1D
                if weight_tensor.dim() == 1:
                    index = tensor_index[0] if isinstance(tensor_index, tuple) else tensor_index
                    if index >= weight_tensor.shape[0]:
                        print(f"ERROR: Tensor index {tensor_index} is out of range for layer '{layer_name}'.")
                        return
                else:
                    index = tensor_index
                    if any(idx >= dim for idx, dim in zip(index, weight_tensor.shape)):
                        print(f"ERROR: Tensor index {tensor_index} is out of range for layer '{layer_name}'.")
                        return

                # ✅ Store the golden value before modification
                if (layer_name, index) not in self.golden_values:
                    self.golden_values[(layer_name, index)] = weight_tensor[index].clone()

                print(f"Accessing layer: {layer_name}")
                print(f"Original weight: {weight_tensor[index].item()}")

                # Convert weight to integer format for bit manipulation
                weight_float = weight_tensor[index].item()
                weight_bytes = struct.pack('f', weight_float)
                weight_int = int.from_bytes(weight_bytes, byteorder='little')

                # ✅ Debug print to check bit before modification
                print(f"Bit {bit} before modification: {bin(weight_int)}")

                if mode == "flip":
                    weight_int ^= (1 << bit)
                elif mode == "stuck":
                    if stuck_value == 1:
                        weight_int |= (1 << bit)
                    else:
                        weight_int &= ~(1 << bit)

                # ✅ Debug print to check bit after modification
                print(f"Bit {bit} after modification: {bin(weight_int)}")

                new_weight_bytes = weight_int.to_bytes(4, byteorder='little')
                new_weight_float = struct.unpack('f', new_weight_bytes)[0]

                weight_tensor[index] = torch.tensor(new_weight_float, dtype=weight_tensor.dtype, device=weight_tensor.device)

                # ✅ Re-pack the updated weights
                packed_params_list = [
                    torch.quantize_per_tensor(weight_tensor, packed_params[1], packed_params[2], packed_params[3])
                ] + list(packed_params[1:])

                state_dict[layer_name] = tuple(packed_params_list)

                self.network.load_state_dict(state_dict)

                print(f"Modified weight: {weight_tensor[index].item()}")

        except AttributeError:
            print(f"ERROR: Layer '{layer_name}' not found in the network.")
        except IndexError:
            print(f"ERROR: Tensor index {tensor_index} is out of range for layer '{layer_name}'.")
        except Exception as e:
            print(f"Unexpected error: {e}")



    def restore_golden(self):
        if not self.golden_values:
            print('CRITICAL ERROR: No golden values stored, skipping restore.')
            return

        state_dict = self.network.state_dict()

        for (layer_name, tensor_index), golden_value in self.golden_values.items():
            # Ensure correct layer name
            if f"module.{layer_name}.weight" in state_dict:
                layer_name = f"module.{layer_name}"

            if f"{layer_name}.weight" not in state_dict:
                print(f"ERROR: Layer '{layer_name}' not found in the network.")
                continue

            weight_tensor = state_dict[f"{layer_name}.weight"]

            if any(index >= dim for index, dim in zip(tensor_index, weight_tensor.shape)):
                print(f"ERROR: Tensor index {tensor_index} is out of range for layer '{layer_name}'.")
                continue

            # Restore the golden value
            weight_tensor[tensor_index] = golden_value

            self.network.load_state_dict(state_dict)
            print(f"Restored weight at {tensor_index} in layer '{layer_name}' to {golden_value.item()}")

        # Clear golden values after restoring
        self.golden_values.clear()
