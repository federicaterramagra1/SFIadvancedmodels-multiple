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
        :param fault_mode: The type of fault to inject (e.g., 'stuck-at' or 'bit-flip').
        """
        for fault in faults:
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
        """
        Inject a bit-flip in the specified layer at the tensor_index position for the specified bit.
        :param layer_name: The name of the layer
        :param tensor_index: The index of the weight to fault inside the tensor
        :param bit: The bit where to inject the fault (0-7 for 8-bit integers)
        """
        self.__int8_bit_flip(layer_name, tensor_index, bit)

    def inject_stuck_at(self, layer_name: str, tensor_index: tuple, bit: int, value: int):
        """
        Inject a stuck-at fault to the specified value in the specified layer at the tensor_index position for the
        specified bit.
        :param layer_name: The name of the layer
        :param tensor_index: The index of the weight to fault inside the tensor
        :param bit: The bit where to inject the fault (0-7 for 8-bit integers)
        :param value: The stuck-at value to set (0 or 1)
        """
        self.__int8_stuck_at(layer_name, tensor_index, bit, value)

    def __int8_bit_flip(self, layer_name: str, tensor_index: tuple, bit: int):
        """
        Inject a bit-flip fault into the weights of the network.
        :param layer_name: The name of the layer
        :param tensor_index: The index of the weight to fault inside the tensor
        :param bit: The bit where to inject the fault (0-7 for 8-bit integers)
        """
        with torch.no_grad():
            # Access the layer
            layer = getattr(self.network.module, layer_name)
            weight_tensor = layer.weight.data.view(torch.uint8)

            # Flip the specified bit
            weight_tensor[tensor_index] = weight_tensor[tensor_index] ^ (1 << bit)

            # Convert back to the original dtype
            layer.weight.data = weight_tensor.view(layer.weight.data.dtype)

    def __int8_stuck_at(self, layer_name: str, tensor_index: tuple, bit: int, value: int):
        """
        Inject a stuck-at fault into the weights of the network.
        :param layer_name: The name of the layer
        :param tensor_index: The index of the weight to fault inside the tensor
        :param bit: The bit where to inject the fault (0-7 for 8-bit integers)
        :param value: The stuck-at value to set (0 or 1)
        """
        with torch.no_grad():
            # Access the layer
            layer = getattr(self.network.module, layer_name)
            weight_tensor = layer.weight.data.view(torch.uint8)

            # Set the bit to the specified value
            if value == 1:
                weight_tensor[tensor_index] = weight_tensor[tensor_index] | (1 << bit)
            else:
                weight_tensor[tensor_index] = weight_tensor[tensor_index] & ~(1 << bit)

            # Convert back to the original dtype
            layer.weight.data = weight_tensor.view(layer.weight.data.dtype)

    def restore_golden(self):
        """
        Restore the value of the faulted network weight to its golden value.
        """
        if self.layer_name is None:
            print('CRITICAL ERROR: impossible to restore the golden value before setting a fault')
            quit()

        self.network.state_dict()[self.layer_name][self.tensor_index] = self.golden_value