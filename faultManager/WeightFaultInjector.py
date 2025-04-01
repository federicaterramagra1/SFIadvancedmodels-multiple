import struct
import torch
import SETTINGS

class WeightFaultInjector:
    def __init__(self, network):
        self.network = network.module if hasattr(network, 'module') else network
        self.golden_values = {}  # Conserva il valore originale (golden) per ogni peso
        self.has_restored = False  # Indica se è già stato fatto il restore
        self.modified_weights = {}  # Contatore di iniezioni per ogni peso

    def inject_faults(self, faults, fault_mode='bit-flip'):
        """Inietta una serie di fault nella rete."""
        for fault in faults:
            # Assicuriamoci di avere il nome corretto del layer
            fault.layer_name = fault.layer_name.replace('module.', '')
            
            # Se non abbiamo ancora salvato il valore originale, lo memorizziamo una volta sola
            if (fault.layer_name, fault.tensor_index) not in self.golden_values:
                self._store_golden_value(fault.layer_name, fault.tensor_index)

            if fault_mode == 'stuck-at_params':
                self.inject_stuck_at(fault.layer_name, fault.tensor_index, fault.bits, fault.value)
            elif fault_mode == 'bit-flip':
                self.inject_bit_flip(fault.layer_name, fault.tensor_index, fault.bits)
            else:
                raise ValueError(f'Fault mode {fault_mode} non valido.')

    def _store_golden_value(self, layer_name, tensor_index):
        """Memorizza il valore originale del peso (golden) prima della prima modifica."""
        if (layer_name, tensor_index) not in self.golden_values:
            state_dict = self.network.state_dict()
            # Recupera il tensore dei pesi quantizzati, dequantizza e salva il valore originale
            weight_tensor = state_dict[f"{layer_name}._packed_params._packed_params"][0].dequantize()
            self.golden_values[(layer_name, tensor_index)] = weight_tensor[tensor_index].clone()

    def inject_bit_flip(self, layer_name: str, tensor_index: tuple, bits: list):
        """Inietta fault di tipo bit-flip sul peso specificato."""
        self._modify_bit(layer_name, tensor_index, bits, mode="flip")

    def inject_stuck_at(self, layer_name: str, tensor_index: tuple, bits: list, value: int):
        """Inietta fault di tipo stuck-at sul peso specificato."""
        self._modify_bit(layer_name, tensor_index, bits, mode="stuck", stuck_value=value)

    def _modify_bit(self, layer_name, tensor_index, bits, mode="flip", stuck_value=None):
        bits = [b for b in bits if b < 4]  # puoi lasciare >= 0 o >= 4 se vuoi solo i bit alti
        if not bits:
            return

        try:
            with torch.no_grad():
                state_dict = self.network.state_dict()
                weight_tensor = state_dict[f"{layer_name}._packed_params._packed_params"][0]
                quantized_value = weight_tensor[tensor_index].int_repr().item()

                print(f" BEFORE: {layer_name}[{tensor_index}] = {quantized_value} ({bin(quantized_value)})")

                for bit in bits:
                    if mode == "flip":
                        quantized_value ^= (1 << bit)
                    elif mode == "stuck":
                        if stuck_value == 1:
                            quantized_value |= (1 << bit)
                        else:
                            quantized_value &= ~(1 << bit)

                quantized_value = max(-128, min(127, quantized_value))
                print(f" AFTER:  {layer_name}[{tensor_index}] = {quantized_value} ({bin(quantized_value)})")
                
                new_quant_tensor = torch.quantize_per_tensor(
                    torch.tensor([quantized_value], dtype=torch.float32),
                    scale=weight_tensor.q_scale(),
                    zero_point=weight_tensor.q_zero_point(),
                    dtype=torch.qint8
                )
                weight_tensor[tensor_index] = new_quant_tensor.dequantize()
                self.network.load_state_dict(state_dict)
                
                print(f" Dequantized value: {new_quant_tensor.dequantize().item()} | Scale: {weight_tensor.q_scale()}, Zero Point: {weight_tensor.q_zero_point()}")

        except Exception as e:
            print(f" ERROR modifying {layer_name}, index {tensor_index}: {e}")


    def restore_golden(self):
        """Ripristina tutti i pesi modificati al loro valore originale."""
        if not self.golden_values:
            print("⚠ CRITICAL ERROR: No golden values stored, skipping restore.")
            return

        state_dict = self.network.state_dict()
        total_restored = 0

        print(f"🔄 Restoring {len(self.golden_values)} golden values...")

        for (layer_name, tensor_index), golden_value in self.golden_values.items():
            key = f"{layer_name}._packed_params._packed_params"
            if key in state_dict:
                weight_tensor = state_dict[key][0].dequantize()
            else:
                print(f" ERROR: Layer '{layer_name}' not found in state_dict.")
                continue

            scale = state_dict[f"{layer_name}.scale"]
            zero_point = state_dict[f"{layer_name}.zero_point"].item()
            quantized_weight = torch.quantize_per_tensor(golden_value.clone(), scale, zero_point, torch.qint8)
            weight_tensor[tensor_index] = quantized_weight.dequantize()
            total_restored += 1

        self.network.load_state_dict(state_dict)
        print(f"✅ Successfully restored {total_restored} golden values.")
        self.golden_values.clear()
