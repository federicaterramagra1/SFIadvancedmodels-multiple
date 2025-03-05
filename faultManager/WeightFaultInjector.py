import torch
import struct

class WeightFaultInjector:

    def __init__(self, network):
        self.network = network
        self.faults = []  # Store multiple faults

    def inject_faults(self, faults):
        """
        Inject multiple faults at once.
        :param faults: List of faults. Each fault is a dictionary with layer_name, tensor_index, bit, and optionally value.
        """
        for fault in faults:
            layer_name = fault["layer_name"]
            tensor_index = fault["tensor_index"]
            bit = fault["bit"]
            value = fault.get("value", None)  # None means bit-flip, otherwise it's stuck-at

            self.__inject_fault(layer_name, tensor_index, bit, value)

    def __inject_fault(self, layer_name, tensor_index, bit, value=None):
        layer = getattr(self.network, layer_name)

        # Extract quantized weight
        weight_q = layer.weight()
        weight_dq = weight_q.dequantize()  # Dequantize to FP32
        scale, zero_point = weight_q.q_scale(), weight_q.q_zero_point()

        # Convert to int8 representation
        weight_int8 = torch.round(weight_dq / scale + zero_point).to(torch.int8)

        self.golden_value = weight_int8[tensor_index].item()  # Store original int8 value

        # Apply bit-flip or stuck-at fault
        if value is None:
            faulty_value = self.__int8_bit_flip(self.golden_value, bit)
        else:
            faulty_value = self.__int8_stuck_at(self.golden_value, bit, value)

        # Update the weight tensor
        weight_int8[tensor_index] = faulty_value

        # Re-quantize and assign back
        weight_dq_faulty = (weight_int8.to(torch.float32) - zero_point) * scale
        layer.set_weight_bias(torch.quantize_per_tensor(weight_dq_faulty, scale, zero_point, torch.qint8), layer.bias())

    def __int8_bit_flip(self, value, bit):
        """
        Flip a bit in an 8-bit integer.
        :param value: Original 8-bit integer
        :param bit: Position of the bit to flip (0-7)
        """
        return value ^ (1 << bit)  # XOR to flip the bit

    def __int8_stuck_at(self, value, bit, set_value):
        """
        Inject a stuck-at fault in an 8-bit integer.
        :param value: Original 8-bit integer
        :param bit: Position of the bit (0-7)
        :param set_value: 0 or 1 (value to "stick" the bit to)
        """
        if set_value == 1:
            return value | (1 << bit)  # Force bit to 1
        else:
            return value & ~(1 << bit)  # Force bit to 0

    def restore_golden(self, layer_name, tensor_index):
        """
        Restore the original value of a weight.
        :param layer_name: Name of the layer
        :param tensor_index: Index of the weight
        """
        if self.golden_value is None:
            print("No fault injected yet.")
            return

        layer = getattr(self.network, layer_name)
        weight_q = layer.weight()
        scale, zero_point = weight_q.q_scale(), weight_q.q_zero_point()

        # Convert golden value back
        golden_float = (self.golden_value - zero_point) * scale

        # Restore the weight
        weight_dq = weight_q.dequantize()
        weight_dq[tensor_index] = golden_float

        # Re-quantize and assign back
        layer.set_weight_bias(torch.quantize_per_tensor(weight_dq, scale, zero_point, torch.qint8), layer.bias())
