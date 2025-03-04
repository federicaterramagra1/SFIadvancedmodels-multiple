import struct
import torch

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
        if fault_mode == 'stuck-at':
            self.__int8_stuck_at(faults)
        elif fault_mode == 'bit-flip':
            self.__int8_bit_flip(faults)
        else:
            raise ValueError(f'Invalid fault mode {fault_mode}')


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

    def __int8_bit_flip(self, faults):
        """
        Inject multiple bit-flip faults into the weights of the network.
        :param faults: List of faults to inject.
        """
        with torch.no_grad():
            for fault in faults:
                # Access the layer through the `module` attribute of the QuantWrapper
                fault.layer_name = f'module.{fault.layer_name}'  # Add 'module.' prefix
                layer = getattr(self.network.module, fault.layer_name)
                weight_tensor = layer.weight.data.view(torch.uint8)
                # Flip the specified bit
                weight_tensor[fault.tensor_index] = weight_tensor[fault.tensor_index] ^ (1 << fault.bit)
                # Convert back to the original dtype
                layer.weight.data = weight_tensor.view(layer.weight.data.dtype)

    def __int8_stuck_at(self, faults):
        """
        Inject multiple stuck-at faults into the weights of the network.
        :param faults: List of faults to inject.
        """
        with torch.no_grad():
            for fault in faults:
                fault.layer_name = f'module.{fault.layer_name}'  # Add 'module.' prefix
                layer = getattr(self.network, fault.layer_name)
                weight_tensor = layer.weight.data.view(torch.uint8)
                # Set the bit to the specified value
                if fault.value == 1:
                    weight_tensor[fault.tensor_index] = weight_tensor[fault.tensor_index] | (1 << fault.bit)
                else:
                    weight_tensor[fault.tensor_index] = weight_tensor[fault.tensor_index] & ~(1 << fault.bit)
                # Convert back to the original dtype
                layer.weight.data = weight_tensor.view(layer.weight.data.dtype)

    def restore_golden(self):
        """
        Restore the value of the faulted network weight to its golden value.
        """
        if self.layer_name is None:
            print('CRITICAL ERROR: impossible to restore the golden value before setting a fault')
            quit()


    def inject_bit_flip(self, layer_name: str, tensor_index: tuple, bit: int):
        """
        Inject a bit-flip in the specified layer at the tensor_index position for the specified bit.
        :param layer_name: The name of the layer
        :param tensor_index: The index of the weight to fault inside the tensor
        :param bit: The bit where to inject the fault (0-7 for 8-bit integers)
        """
        self.__inject_fault(layer_name=layer_name, tensor_index=tensor_index, bit=bit)

    def inject_stuck_at(self, layer_name: str, tensor_index: tuple, bit: int, value: int):
        """
        Inject a stuck-at fault to the specified value in the specified layer at the tensor_index position for the
        specified bit.
        :param layer_name: The name of the layer
        :param tensor_index: The index of the weight to fault inside the tensor
        :param bit: The bit where to inject the fault (0-7 for 8-bit integers)
        :param value: The stuck-at value to set (0 or 1)
        """
        self.__inject_fault(layer_name=layer_name, tensor_index=tensor_index, bit=bit, value=value)