import torch

class WeightFaultInjector:
    def __init__(self, network):
        self.network = network
        self.layer_name = None
        self.tensor_index = None
        self.bit = None
        self.golden_value = None

    def __inject_fault(self, layer_name, tensor_index, bit, value=None):
        """
        Internal method to inject a fault into a quantized weight tensor.
        """
        self.layer_name = layer_name
        self.tensor_index = tensor_index
        self.bit = bit

        # Get the quantized layer
        layer = getattr(self.network, self.layer_name)

        # Dequantize the weight tensor to access its values
        weight_tensor = layer.weight().dequantize()  # Dequantize to floating-point
        self.golden_value = weight_tensor[self.tensor_index].item()  # Save the golden value

        # Apply the fault injection logic
        if value is None:
            faulty_value = self.__int8_bit_flip(self.golden_value, bit)
        else:
            faulty_value = self.__int8_stuck_at(self.golden_value, bit, value)

        # Modify the weight tensor
        weight_tensor[self.tensor_index] = faulty_value

        # Re-quantize the weight tensor and assign it back to the layer
        quantized_weight = torch.quantize_per_tensor(
            weight_tensor, layer.scale, layer.zero_point, torch.qint8
        )
        layer.set_weight_bias(quantized_weight, layer.bias())

    def __int8_bit_flip(self, value, bit):
        """
        Inject a bit-flip on an 8-bit integer value.
        :param value: The original floating-point value.
        :param bit: The bit to flip.
        :return: The faulty floating-point value.
        """
        # Convert the floating-point value to an 8-bit integer
        int8_value = int(value / self.network.state_dict()[self.layer_name + '.scale']) + self.network.state_dict()[self.layer_name + '.zero_point']
        int8_value = int8_value & 0xFF  # Ensure it's 8 bits

        # Flip the specified bit
        faulty_int8_value = int8_value ^ (1 << bit)

        # Convert back to floating-point
        faulty_value = (faulty_int8_value - self.network.state_dict()[self.layer_name + '.zero_point']) * self.network.state_dict()[self.layer_name + '.scale']
        return faulty_value

    def __int8_stuck_at(self, value, bit, stuck_value):
        """
        Inject a stuck-at fault on an 8-bit integer value.
        :param value: The original floating-point value.
        :param bit: The bit to modify.
        :param stuck_value: The value to set (0 or 1).
        :return: The faulty floating-point value.
        """
        # Convert the floating-point value to an 8-bit integer
        int8_value = int(value / self.network.state_dict()[self.layer_name + '.scale']) + self.network.state_dict()[self.layer_name + '.zero_point']
        int8_value = int8_value & 0xFF  # Ensure it's 8 bits

        # Set the specified bit to the stuck-at value
        if stuck_value == 1:
            faulty_int8_value = int8_value | (1 << bit)
        else:
            faulty_int8_value = int8_value & ~(1 << bit)

        # Convert back to floating-point
        faulty_value = (faulty_int8_value - self.network.state_dict()[self.layer_name + '.zero_point']) * self.network.state_dict()[self.layer_name + '.scale']
        return faulty_value

    def restore_golden(self):
        """
        Restore the value of the faulted network weight to its golden value.
        """
        if self.layer_name is None:
            print('CRITICAL ERROR: impossible to restore the golden value before setting a fault')
            quit()

        # Get the quantized layer
        layer = getattr(self.network, self.layer_name)

        # Dequantize the weight tensor
        weight_tensor = layer.weight().dequantize()

        # Restore the golden value
        weight_tensor[self.tensor_index] = self.golden_value

        # Re-quantize the weight tensor and assign it back to the layer
        quantized_weight = torch.quantize_per_tensor(
            weight_tensor, layer.scale, layer.zero_point, torch.qint8
        )
        layer.set_weight_bias(quantized_weight, layer.bias())

    def inject_bit_flip(self, layer_name, tensor_index, bit):
        """
        Inject a bit-flip in the specified layer at the tensor_index position for the specified bit.
        :param layer_name: The name of the layer.
        :param tensor_index: The index of the weight to fault inside the tensor.
        :param bit: The bit where to inject the fault.
        """
        self.__inject_fault(layer_name=layer_name, tensor_index=tensor_index, bit=bit)

    def inject_stuck_at(self, layer_name, tensor_index, bit, value):
        """
        Inject a stuck-at fault to the specified value in the specified layer at the tensor_index position for the
        specified bit.
        :param layer_name: The name of the layer.
        :param tensor_index: The index of the weight to fault inside the tensor.
        :param bit: The bit where to inject the fault.
        :param value: The stuck-at value to set.
        """
        self.__inject_fault(layer_name=layer_name, tensor_index=tensor_index, bit=bit, value=value)

    def inject_faults(self, faults: list, fault_mode='stuck-at'):
        """
        Inject multiple faults into the network.
        :param faults: List of faults to inject.
        :param fault_mode: The type of fault to inject (e.g., 'stuck-at' or 'bit-flip').
        """
        for fault in faults:
            # Handle 'module.' prefix in layer names
            if fault.layer_name.startswith('module.'):
                fault.layer_name = fault.layer_name.replace('module.', '')

            # Inject the fault based on the fault mode
            if fault_mode == 'stuck-at':
                self.inject_stuck_at(fault.layer_name, fault.tensor_index, fault.bit, fault.value)
            elif fault_mode == 'bit-flip':
                self.inject_bit_flip(fault.layer_name, fault.tensor_index, fault.bit)
            else:
                raise ValueError(f"Unsupported fault mode: {fault_mode}")