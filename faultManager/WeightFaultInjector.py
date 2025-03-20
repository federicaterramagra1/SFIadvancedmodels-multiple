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

            if fault_mode == 'stuck-at_params':
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
        """Modifica i bit solo sui valori quantizzati."""
        if self.modified_weights.get((layer_name, tensor_index), 0) >= SETTINGS.NUM_FAULTS_TO_INJECT:
            #print(f" Skipping duplicate modification for {layer_name}[{tensor_index}]")
            return  

        self.modified_weights[(layer_name, tensor_index)] = self.modified_weights.get((layer_name, tensor_index), 0) + 1 

        try:
            with torch.no_grad():
                state_dict = self.network.state_dict()
                
                # Recupera il tensore dei pesi quantizzati
                weight_tensor = state_dict[f"{layer_name}._packed_params._packed_params"][0]
                
                # Ottieni il valore quantizzato come intero
                quantized_value = weight_tensor[tensor_index].int_repr().item()  

                print(f" BEFORE: {layer_name}[{tensor_index}] = {quantized_value} ({bin(quantized_value)})")

                # Bit-flip sugli 8 bit significativi
                for bit in bits:
                    if mode == "flip":
                        quantized_value ^= (1 << bit)  
                    elif mode == "stuck":
                        if stuck_value == 1:
                            quantized_value |= (1 << bit)
                        else:
                            quantized_value &= ~(1 << bit)

                # Mantieni il valore nell'intervallo corretto di quantizzazione (-128 a 127 per qint8)
                quantized_value = max(-128, min(127, quantized_value))

                print(f" AFTER: {layer_name}[{tensor_index}] = {quantized_value} ({bin(quantized_value)})")

                # Aggiorna il tensore con il nuovo valore quantizzato
                weight_tensor[tensor_index] = torch.quantize_per_tensor(
                    torch.tensor([quantized_value], dtype=torch.float32),
                    scale=weight_tensor.q_scale(),
                    zero_point=weight_tensor.q_zero_point(),
                    dtype=torch.qint8
                )

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
            else:
                print(f" ERROR: Layer '{layer_name}' not found in state_dict.")
                continue

            # Re-quantizza il peso originale
            scale = state_dict[f"{layer_name}.scale"]
            zero_point = state_dict[f"{layer_name}.zero_point"].item()
            quantized_weight = torch.quantize_per_tensor(golden_value.clone(), scale, zero_point, torch.qint8)

            weight_tensor[tensor_index] = quantized_weight.dequantize()
            total_restored += 1

        self.network.load_state_dict(state_dict)
        print(f"✅ Successfully restored {total_restored} golden values.")
        self.golden_values.clear()
