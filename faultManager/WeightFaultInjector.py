import struct


class WeightFaultInjector:
    def __init__(self, network):
        self.network = network

        self.layer_name = None
        self.tensor_index = None
        self.bit = None

        self.golden_value = None

    def __inject_fault(self, layer_name, tensor_index, bit, value=None):
        self.layer_name = layer_name
        self.tensor_index = tensor_index
        self.bit = bit

        # Get the golden value as an 8-bit integer
        self.golden_value = int(self.network.state_dict()[self.layer_name][self.tensor_index])

        # If the value is not set, then we are doing a bit-flip
        if value is None:
            faulty_value = self.__int8_bit_flip()
        else:
            faulty_value = self.__int8_stuck_at(value)

        self.faulty_value = faulty_value

        # Update the weight with the faulty value
        self.network.state_dict()[self.layer_name][self.tensor_index] = faulty_value

    def __int8_bit_flip(self):
        """
        Inject a bit-flip on an 8-bit integer.
        :return: The value of the bit-flip on the golden value
        """
        # Perform bit-flip using XOR with a bitmask
        bitmask = 1 << self.bit  # Create a bitmask for the specified bit
        faulted_value = self.golden_value ^ bitmask  # Apply XOR to flip the bit
        return faulted_value

    def __int8_stuck_at(self, value: int):
        """
        Inject a stuck-at fault on an 8-bit integer.
        :param value: The value to use as stuck-at value (0 or 1)
        :return: The value of the stuck-at fault on the golden value
        """
        bitmask = 1 << self.bit  # Create a bitmask for the specified bit
        if value == 1:
            # Set the bit to 1 using OR
            faulted_value = self.golden_value | bitmask
        else:
            # Set the bit to 0 using AND with the inverted bitmask
            faulted_value = self.golden_value & ~bitmask
        return faulted_value

    def restore_golden(self):
        """
        Restore the value of the faulted network weight to its golden value.
        """
        if self.layer_name is None:
            print('CRITICAL ERROR: impossible to restore the golden value before setting a fault')
            quit()

        self.network.state_dict()[self.layer_name][self.tensor_index] = self.golden_value

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