import struct
import torch
import copy

class WeightFaultInjector:
    def __init__(self, network):
        self.network = network.module if hasattr(network, 'module') else network
        self.golden_parameters = copy.deepcopy(network.state_dict())
        self.layer_name = None
        self.tensor_index = None
        self.bit = None
        self.golden_value = None

    def inject_faults(self, faults: list, fault_mode='stuck-at'):
        for fault in faults:
            if fault.layer_name.startswith('module.'):
                fault.layer_name = fault.layer_name.replace('module.', '')
            print(f"Injecting fault in layer: {fault.layer_name}")  
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

                if any(index >= dim for index, dim in zip(tensor_index, weight_tensor.shape)):
                    print(f"ERROR: Tensor index {tensor_index} is out of range for layer '{layer_name}'.")
                    return

                print(f"Accessing layer: {layer_name}")  
                print(f"Original weight: {weight_tensor[tensor_index].item()}")  
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

                self.golden_value = weight_tensor[tensor_index].clone()
                weight_tensor[tensor_index] = torch.tensor(new_weight_float, dtype=weight_tensor.dtype, device=weight_tensor.device)
                
                if f"{layer_name}._packed_params._packed_params" in state_dict:
                    scales = packed_params[1]
                    zero_points = packed_params[2]
                    axis = packed_params[3]
                    dtype = packed_params[4]
                    packed_params_list = [
                        torch.quantize_per_channel(weight_tensor, scales, zero_points, axis, dtype),
                        scales, zero_points, axis, dtype
                    ]
                    state_dict[f"{layer_name}._packed_params._packed_params"] = tuple(packed_params_list)
                else:
                    state_dict[f"{layer_name}.weight"] = weight_tensor
                
                self.network.load_state_dict(state_dict)
                print(f"Modified weight: {weight_tensor[tensor_index].item()}")  
        except AttributeError:
            print(f"ERROR: Layer '{layer_name}' not found in the network.")
        except IndexError:
            print(f"ERROR: Tensor index {tensor_index} is out of range for layer '{layer_name}'.")
        except Exception as e:
            print(f"Unexpected error: {e}")

    def restore_golden(self):
        for name, param in self.network.named_parameters():
            if 'packed_params' in name:
                try:
                    packed_params = self.golden_parameters[name]
                    weights, biases, zero_points, scales = packed_params
                    param.data.copy_(weights)
                    if param.bias is not None:
                        param.bias.copy_(biases)
                except IndexError:
                    print(f"Warning: Packed parameters for layer '{name}' do not have the expected format. Skipping restore for this layer.")
                    continue
            else:
                param.data.copy_(self.golden_parameters[name])
