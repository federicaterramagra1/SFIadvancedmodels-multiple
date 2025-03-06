class WeightFaultInjector:
    def __init__(self, network):
        self.network = network.module if hasattr(network, 'module') else network
        self.golden_values = {}  # Dictionary to store golden values for each fault

    def inject_faults(self, faults: list, fault_mode='stuck-at'):
        for fault in faults:
            if fault.layer_name.startswith('module.'):
                fault.layer_name = fault.layer_name.replace('module.', '')
            print(f"Injecting fault in layer: {fault.layer_name}")
            self.inject_fault(fault, fault_mode)

    def inject_fault(self, fault, fault_mode='stuck-at'):
        layer_name = fault.layer_name
        tensor_index = fault.tensor_index
        bit = fault.bit
        key = (layer_name, tensor_index, bit)  # Unique key for each fault

        if fault_mode == 'stuck-at':
            self.inject_stuck_at(layer_name, tensor_index, bit, fault.value, key)
        elif fault_mode == 'bit-flip':
            self.inject_bit_flip(layer_name, tensor_index, bit, key)
        else:
            raise ValueError(f'Invalid fault mode {fault_mode}')

    def inject_bit_flip(self, layer_name: str, tensor_index: tuple, bit: int, key: tuple):
        self.__modify_bit(layer_name, tensor_index, bit, mode="flip", key=key)

    def inject_stuck_at(self, layer_name: str, tensor_index: tuple, bit: int, value: int, key: tuple):
        self.__modify_bit(layer_name, tensor_index, bit, mode="stuck", stuck_value=value, key=key)

    def __modify_bit(self, layer_name: str, tensor_index: tuple, bit: int, mode="flip", stuck_value=None, key=None):
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

                # Store the golden value if not already stored
                if key not in self.golden_values:
                    self.golden_values[key] = weight_tensor[tensor_index].clone()

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
        if not self.golden_values:
            print('CRITICAL ERROR: impossible to restore the golden value before setting a fault')
            return
        state_dict = self.network.state_dict()
        for key, golden_value in self.golden_values.items():
            layer_name, tensor_index, _ = key
            if f"{layer_name}._packed_params._packed_params" in state_dict:
                packed_params = state_dict[f"{layer_name}._packed_params._packed_params"]
                weight_tensor = packed_params[0].dequantize()
            else:
                weight_tensor = state_dict[f"{layer_name}.weight"]

            if any(index >= dim for index, dim in zip(tensor_index, weight_tensor.shape)):
                print(f"ERROR: Tensor index {tensor_index} is out of range for layer '{layer_name}'.")
                continue

            weight_tensor[tensor_index] = golden_value
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
            print(f"Restored weight to golden value: {weight_tensor[tensor_index].item()}")