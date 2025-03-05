import torch
import struct

class WeightFaultInjector:
    def __init__(self, network):
        self.network = network
        self.layer_name = None
        self.tensor_index = None
        self.bit = None
        self.golden_value = None

    def inject_faults(self, faults: list, fault_mode='stuck-at'):
        """
        Inject multiple faults into the network.
        :param faults: List of faults to inject.
        :param fault_mode: The type of fault to inject ('stuck-at' or 'bit-flip').
        """
        for fault in faults:
            if fault.layer_name.startswith('module.'):
                fault.layer_name = fault.layer_name.replace('module.', '')  # Remove 'module.' prefix
            self.inject_fault(fault, fault_mode)

    def inject_fault(self, fault, fault_mode='stuck-at'):
        """
        Inject a single fault into the network.
        """
        if fault_mode == 'stuck-at':
            self.inject_stuck_at(fault.layer_name, fault.tensor_index, fault.bit, fault.value)
        elif fault_mode == 'bit-flip':
            self.inject_bit_flip(fault.layer_name, fault.tensor_index, fault.bit)
        else:
            raise ValueError(f'Invalid fault mode {fault_mode}')

    def inject_bit_flip(self, layer_name: str, tensor_index: tuple, bit: int):
        """ Inject a bit-flip at the given location. """
        self.__modify_bit(layer_name, tensor_index, bit, mode="flip")

    def inject_stuck_at(self, layer_name: str, tensor_index: tuple, bit: int, value: int):
        """ Inject a stuck-at fault at the given location. """
        self.__modify_bit(layer_name, tensor_index, bit, mode="stuck", stuck_value=value)

    def __modify_bit(self, layer_name: str, tensor_index: tuple, bit: int, mode="flip", stuck_value=None):
        """
        Modify a specific bit in the tensor (bit-flip or stuck-at).
        """
        try:
            with torch.no_grad():
                # Access the layer, handling DataParallel if needed
                layer = getattr(self.network.module, layer_name) if hasattr(self.network, "module") else getattr(self.network, layer_name)

                # Convert weights to uint8 for bit manipulation
                weight_tensor = layer.weight.data
                weight_float = weight_tensor[tensor_index].item()
                
                # Convert float to bytes (IEEE 754) for bit-level modification
                weight_bytes = struct.pack('f', weight_float)
                weight_int = int.from_bytes(weight_bytes, byteorder='little')

                if mode == "flip":
                    weight_int ^= (1 << bit)  # Bit-flip
                elif mode == "stuck":
                    if stuck_value == 1:
                        weight_int |= (1 << bit)  # Force bit to 1
                    else:
                        weight_int &= ~(1 << bit)  # Force bit to 0

                # Convert back to float
                new_weight_bytes = weight_int.to_bytes(4, byteorder='little')
                new_weight_float = struct.unpack('f', new_weight_bytes)[0]

                # Assign modified weight back to tensor
                layer.weight.data[tensor_index] = torch.tensor(new_weight_float, dtype=weight_tensor.dtype, device=weight_tensor.device)

        except AttributeError:
            print(f"ERROR: Layer '{layer_name}' not found in the network.")
        except IndexError:
            print(f"ERROR: Tensor index {tensor_index} is out of range for layer '{layer_name}'.")
        except Exception as e:
            print(f"Unexpected error: {e}")

    def restore_golden(self):
        """
        Restore the value of the faulted network weight to its golden value.
        """
        if self.layer_name is None:
            print('CRITICAL ERROR: impossible to restore the golden value before setting a fault')
            quit()

        self.network.state_dict()[self.layer_name][self.tensor_index] = self.golden_value
