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
          
          # Check if the layer has _packed_params
          if hasattr(layer, '_packed_params'):
              # Unpack the weights
              weight_tensor = layer._packed_params._packed_params[0]  # Access the packed weights
              weight_tensor = weight_tensor.dequantize()  # Dequantize to get the full precision tensor
              weight_tensor = weight_tensor.view(torch.uint8)  # Convert to uint8 for bit manipulation
          else:
              # If not quantized, access the weights directly
              weight_tensor = layer.weight.data.view(torch.uint8)

          # Flip the specified bit
          weight_tensor[tensor_index] = weight_tensor[tensor_index] ^ (1 << bit)

          # Convert back to the original dtype
          if hasattr(layer, '_packed_params'):
              # Re-quantize the weights
              weight_tensor = weight_tensor.view(layer.weight.dtype)  # Convert back to the original dtype
              layer._packed_params._packed_params[0] = torch.quantize_per_tensor(weight_tensor, scale=layer.scale, zero_point=layer.zero_point, dtype=torch.qint8)
          else:
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
          print(f"Fault layer name: {layer_name}")  # Debugging
          print(f"Tensor index: {tensor_index}")  # Debugging
          print(f"Bit: {bit}")  # Debugging
          print(f"Value: {value}")  # Debugging
          layer = getattr(self.network.module, layer_name)
          
          # Check if the layer has _packed_params
          if hasattr(layer, '_packed_params'):
              # Unpack the weights
              weight_tensor = layer._packed_params._packed_params[0]  # Access the packed weights
              weight_tensor = weight_tensor.dequantize()  # Dequantize to get the full precision tensor
              weight_tensor = weight_tensor.view(torch.uint8)  # Convert to uint8 for bit manipulation
          else:
              # If not quantized, access the weights directly
              weight_tensor = layer.weight.data.view(torch.uint8)

          # Set the bit to the specified value
          if value == 1:
              weight_tensor[tensor_index] = weight_tensor[tensor_index] | (1 << bit)
          else:
              weight_tensor[tensor_index] = weight_tensor[tensor_index] & ~(1 << bit)

          # Convert back to the original dtype
          if hasattr(layer, '_packed_params'):
              # Re-quantize the weights
              weight_tensor = layer._packed_params.get_weight()  # Convert back to the original dtype
              layer._packed_params._packed_params[0] = torch.quantize_per_tensor(weight_tensor, scale=layer.scale, zero_point=layer.zero_point, dtype=torch.qint8)
          else:
              layer.weight.data = weight_tensor.view(layer.weight.data.dtype)

    def restore_golden(self):
        """
        Restore the value of the faulted network weight to its golden value.
        """
        if self.layer_name is None:
            print('CRITICAL ERROR: impossible to restore the golden value before setting a fault')
            quit()

        self.network.state_dict()[self.layer_name][self.tensor_index] = self.golden_value