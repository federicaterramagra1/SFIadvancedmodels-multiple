import struct
import torch

class WeightFaultInjector:
    def __init__(self, network):
        self.network = network.module if hasattr(network, 'module') else network
        self.layer_name = None
        self.tensor_index = None
        self.bit = None
        self.golden_value = None

    def inject_faults(self, faults: list, fault_mode='stuck-at'):
        for fault in faults:
            if fault.layer_name.startswith('module.'):
                fault.layer_name = fault.layer_name.replace('module.', '')
            print(f"Injecting fault in layer: {fault.layer_name}")  # Debug print
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
        self.__modify_bit(layer_name, tensor_index, bit, mode="flip")

    def inject_stuck_at(self, layer_name: str, tensor_index: tuple, bit: int, value: int):
        self.__modify_bit(layer_name, tensor_index, bit, mode="stuck", stuck_value=value)

    def __modify_bit(self, layer_name: str, tensor_index: tuple, bit: int, mode="flip", stuck_value=None):
        try:
            with torch.no_grad():
                state_dict = self.network.state_dict()
                if f"{layer_name}._packed_params._packed_params" in state_dict:
                    packed_params = state_dict[f"{layer_name}._packed_params._packed_params"]
                    weight_tensor = packed_params[0].dequantize()
                else:
                    weight_tensor = state_dict[f"{layer_name}.weight"]

                # Check if tensor index is within bounds
                if any(index >= dim for index, dim in zip(tensor_index, weight_tensor.shape)):
                    print(f"ERROR: Tensor index {tensor_index} is out of range for layer '{layer_name}'.")
                    return

                print(f"Accessing layer: {layer_name}")  # Debug print
                print(f"Original weight: {weight_tensor[tensor_index]}")  # Debug print
                weight_float = weight_tensor[tensor_index].item()
                weight_bytes = struct.pack('f', weight_float)
                weight_int = int.from_bytes(weight_bytes, byteorder='little')

                if mode == "flip":
                    weight_int ^= (1 << bit)
                elif mode == "stuck":
                    if stuck_value == 1:
                        weight_int |= (1 << bit)
                    else:
                        weight_int &= ~(1 << bit)

                new_weight_bytes = weight_int.to_bytes(4, byteorder='little')
                new_weight_float = struct.unpack('f', new_weight_bytes)[0]

                self.golden_value = weight_tensor[tensor_index].clone()  # Store golden value
                weight_tensor[tensor_index] = torch.tensor(new_weight_float, dtype=weight_tensor.dtype, device=weight_tensor.device)
                
                if f"{layer_name}._packed_params._packed_params" in state_dict:
                    # Re-quantize the weights
                    packed_params[0] = torch.quantize_per_channel(weight_tensor, scales=packed_params[1], zero_points=packed_params[2], axis=packed_params[3], dtype=packed_params[4])
                    state_dict[f"{layer_name}._packed_params._packed_params"] = packed_params
                else:
                    state_dict[f"{layer_name}.weight"] = weight_tensor
                
                self.network.load_state_dict(state_dict)
                print(f"Modified weight: {weight_tensor[tensor_index]}")  # Debug print
        except AttributeError:
            print(f"ERROR: Layer '{layer_name}' not found in the network.")
        except IndexError:
            print(f"ERROR: Tensor index {tensor_index} is out of range for layer '{layer_name}'.")
        except Exception as e:
            print(f"Unexpected error: {e}")

    def restore_golden(self):
        if self.layer_name is None or self.tensor_index is None or self.golden_value is None:
            print('CRITICAL ERROR: impossible to restore the golden value before setting a fault')
            return
        state_dict = self.network.state_dict()
        if f"{self.layer_name}._packed_params._packed_params" in state_dict:
            packed_params = state_dict[f"{self.layer_name}._packed_params._packed_params"]
            weight_tensor = packed_params[0].dequantize()
        else:
            weight_tensor = state_dict[f"{self.layer_name}.weight"]
        
        # Check if tensor index is within bounds
        if any(index >= dim for index, dim in zip(self.tensor_index, weight_tensor.shape)):
            print(f"ERROR: Tensor index {self.tensor_index} is out of range for layer '{self.layer_name}'.")
            return
        
        weight_tensor[self.tensor_index] = self.golden_value
        if f"{self.layer_name}._packed_params._packed_params" in state_dict:
            packed_params[0] = torch.quantize_per_channel(weight_tensor, scales=packed_params[1], zero_points=packed_params[2], axis=packed_params[3], dtype=packed_params[4])
            state_dict[f"{self.layer_name}._packed_params._packed_params"] = packed_params
        else:
            state_dict[f"{self.layer_name}.weight"] = weight_tensor
        self.network.load_state_dict(state_dict)
        print(f"Restored weight to golden value: {self.golden_value}")  # Debug print
