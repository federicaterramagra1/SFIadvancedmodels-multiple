import struct
import torch
import SETTINGS

class WeightFaultInjector:
    def __init__(self, network):
        self.network = network.module if hasattr(network, 'module') else network
        self.golden_values = {}  # Store original weights before modification
        self.has_restored = False  # Track if restore_golden() was already executed
        self.modified_weights = {}  # Track modified weights

    def inject_faults(self, faults, fault_mode='bit-flip'):
        """Inject multiple faults into the network"""
        for fault in faults:
            fault.layer_name = fault.layer_name.replace('module.', '')  # Ensure correct layer name format
            
            if (fault.layer_name, fault.tensor_index) not in self.golden_values:
                self._store_golden_value(fault.layer_name, fault.tensor_index)

            if fault_mode == 'stuck-at':
                self.inject_stuck_at(fault.layer_name, fault.tensor_index, fault.bits, fault.value)
            elif fault_mode == 'bit-flip':
                self.inject_bit_flip(fault.layer_name, fault.tensor_index, fault.bits)
            else:
                raise ValueError(f'Invalid fault mode {fault_mode}')


    def _store_golden_value(self, layer_name, tensor_index):
        """Ensure all golden values are stored before modification"""
        if (layer_name, tensor_index) not in self.golden_values:
            state_dict = self.network.state_dict()
            weight_tensor = state_dict[f"{layer_name}._packed_params._packed_params"][0].dequantize()
            self.golden_values[(layer_name, tensor_index)] = weight_tensor[tensor_index].clone()


    def inject_bit_flip(self, layer_name: str, tensor_index: tuple, bits: list):
        """Inject bit-flip faults into specified weight"""
        self._modify_bit(layer_name, tensor_index, bits, mode="flip")

    def inject_stuck_at(self, layer_name: str, tensor_index: tuple, bits: list, value: int):
        """Inject stuck-at faults into specified weight"""
        self._modify_bit(layer_name, tensor_index, bits, mode="stuck", stuck_value=value)


    def _modify_bit(self, layer_name, tensor_index, bits, mode="flip", stuck_value=None):
        """Modify specified bits in the weight"""
        if self.modified_weights.get((layer_name, tensor_index), 0) >= SETTINGS.NUM_FAULTS_TO_INJECT:
            print(f"⚠️ Skipping duplicate modification for {layer_name}[{tensor_index}]")
            return  # Skip extra modifications beyond NUM_FAULTS_TO_INJECT
        self.modified_weights[(layer_name, tensor_index)] = self.modified_weights.get((layer_name, tensor_index), 0) + 1 # Mark as modified

        try:
            with torch.no_grad():
                state_dict = self.network.state_dict()
                weight_tensor = state_dict[f"{layer_name}._packed_params._packed_params"][0].dequantize()
                original_weight = weight_tensor[tensor_index].item()

                print(f" BEFORE: {layer_name}[{tensor_index}] = {original_weight:.6f}")

                weight_bytes = struct.pack('f', original_weight)
                weight_int = int.from_bytes(weight_bytes, byteorder='little')

                for bit in sorted(bits, reverse=True):  # Flip higher bits first
                    if mode == "flip":
                        weight_int ^= (1 << bit)  
                    elif mode == "stuck":
                        if stuck_value == 1:
                            weight_int |= (1 << bit)  
                        else:
                            weight_int &= ~(1 << bit)

                new_weight_bytes = weight_int.to_bytes(4, byteorder='little')
                new_weight_float = struct.unpack('f', new_weight_bytes)[0]

                print(f" AFTER: {layer_name}[{tensor_index}] = {new_weight_float:.6f}")

                scale = state_dict[f"{layer_name}.scale"]
                zero_point = state_dict[f"{layer_name}.zero_point"].item()
                quantized_weight = torch.quantize_per_tensor(
                    torch.tensor(new_weight_float, dtype=torch.float32), scale, zero_point, torch.qint8
                )

                weight_tensor[tensor_index] = quantized_weight.dequantize()
                self.network.load_state_dict(state_dict)

        except Exception as e:
            print(f" ERROR modifying {layer_name}, index {tensor_index}: {e}")



    def restore_golden(self):
        """Restore all modified weights to their original values"""
        if not self.golden_values:
            if not self.has_restored:
                print("⚠ CRITICAL ERROR: No golden values stored, skipping restore.")
                self.has_restored = True
            return

        state_dict = self.network.state_dict()
        total_restored = 0

        print(f"🔄 Restoring {len(self.golden_values)} golden values...")

        for (layer_name, tensor_index), golden_value in self.golden_values.items():
            if f"{layer_name}._packed_params._packed_params" in state_dict:
                weight_tensor = state_dict[f"{layer_name}._packed_params._packed_params"][0].dequantize()
            elif f"{layer_name}.weight" in state_dict:
                weight_tensor = state_dict[f"{layer_name}.weight"]
            else:
                print(f" ERROR: Layer '{layer_name}' not found in state_dict.")
                continue

            # Re-quantize the restored weight
            scale = state_dict[f"{layer_name}.scale"]
            zero_point = state_dict[f"{layer_name}.zero_point"].item()
            quantized_weight = torch.quantize_per_tensor(golden_value.clone(), scale, zero_point, torch.qint8)
            weight_tensor[tensor_index] = quantized_weight.dequantize()

            total_restored += 1

        # Repack the quantized weights
        for name, module in self.network.named_modules():
            if isinstance(module, torch.ao.nn.quantized.Linear):
                module._packed_params._packed_params = torch.ops.quantized.linear_prepack(module.weight(), module.bias())

        self.network.load_state_dict(state_dict)
        print(f"✅ Successfully restored {total_restored} golden values.")

        # Clear golden values after restore
        self.golden_values.clear()
