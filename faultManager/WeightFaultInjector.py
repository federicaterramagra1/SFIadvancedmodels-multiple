import torch
import numpy as np
import SETTINGS

DEBUG_FI = False   # False per silenziare

class WeightFaultInjector:
    def __init__(self, network):
        # se il modello è wrappato in DataParallel, prendi il modulo interno
        self.network = network.module if hasattr(network, 'module') else network
        # mappa: (layer_name, tensor_index) -> golden_int8
        self.golden_values = {}

    # ---------- helpers compatibilità API quantizzate ----------
    def _get_qweight_and_bias(self, layer):
        """
        Restituisce (quant_weight_tensor, bias) usando le API pubbliche se disponibili,
        altrimenti fa fallback ai packed_params.
        """
        # API pubblica (preferita)
        if hasattr(layer, "weight"):
            try:
                return layer.weight(), layer.bias()
            except Exception:
                pass
        # Fallback su packed_params privati
        if hasattr(layer, "_packed_params") and hasattr(layer._packed_params, "_weight_bias"):
            return layer._packed_params._weight_bias()
        raise RuntimeError("Impossibile ottenere peso/bias quantizzati dal layer.")

    def _set_qweight_and_bias(self, layer, q_weight, bias):
        """
        Imposta (peso, bias) sul layer quantizzato usando l'API disponibile.
        """
        if hasattr(layer, "set_weight_bias"):
            layer.set_weight_bias(q_weight, bias)
            return
        if hasattr(layer, "_packed_params") and hasattr(layer._packed_params, "set_weight_bias"):
            layer._packed_params.set_weight_bias(q_weight, bias)
            return
        raise RuntimeError("Impossibile impostare peso/bias quantizzati sul layer.")

    # -----------------------------------------------------------

    def inject_faults(self, faults, fault_mode='bit-flip'):
        for fault in faults:
            # normalizza il nome (rimuovi 'module.' se presente)
            fault.layer_name = fault.layer_name.replace('module.', '')

            # salva il valore INT originale per quel peso (una sola volta)
            if (fault.layer_name, fault.tensor_index) not in self.golden_values:
                self._store_golden_value(fault.layer_name, fault.tensor_index)

            if fault_mode == 'stuck-at_params':
                # per stuck-at: value=0/1 e lista di bit
                self.inject_stuck_at(fault.layer_name, fault.tensor_index, fault.bits, fault.value)
            elif fault_mode == 'bit-flip':
                self.inject_bit_flip(fault.layer_name, fault.tensor_index, fault.bits)
            else:
                raise ValueError(f'Fault mode {fault_mode} non valido.')

    def _store_golden_value(self, layer_name, tensor_index):
        """
        Salva l'INT8 originale del peso interessato, così il restore è bit-perfetto.
        """
        with torch.no_grad():
            layer = dict(self.network.named_modules())[layer_name]
            wq, _ = self._get_qweight_and_bias(layer)
            # int_repr() restituisce int8; lo salvo come int Python
            golden_int = int(wq.int_repr()[tensor_index].item())
            self.golden_values[(layer_name, tensor_index)] = golden_int

    def inject_bit_flip(self, layer_name: str, tensor_index: tuple, bits: list):
        self._modify_bit(layer_name, tensor_index, bits, mode="flip")

    def inject_stuck_at(self, layer_name: str, tensor_index: tuple, bits: list, value: int):
        # value: 0 (stuck-at-0) oppure 1 (stuck-at-1)
        self._modify_bit(layer_name, tensor_index, bits, mode="stuck", stuck_value=value)

    def _modify_bit(self, layer_name, tensor_index, bits, mode="flip", stuck_value=None):
        bits = [b for b in bits if 0 <= b < 8]
        if not bits:
            return

        try:
            with torch.no_grad():
                layer = dict(self.network.named_modules())[layer_name]
                wq, bias = self._get_qweight_and_bias(layer)

                if wq.qscheme() != torch.per_tensor_affine:
                    raise NotImplementedError("Gestito solo per per-tensor.")

                scale = wq.q_scale()
                zero_point = wq.q_zero_point()

                # --- BEFORE: leggi lo stato originario ---
                ir_before = wq.int_repr().clone()              # int8
                v_before_i8 = int(ir_before[tensor_index].item())
                v_before_u8 = (v_before_i8 + 256) % 256
                deq_before = wq.dequantize()[tensor_index].item()

                # vista uint8 per XOR sicuro
                ir_u8 = ir_before.view(torch.uint8)

                # --- applica modifiche ---
                v_new = v_before_u8
                for bit in bits:
                    if mode == "flip":
                        v_new ^= (1 << bit)
                    elif mode == "stuck":
                        v_new = (v_new | (1 << bit)) if stuck_value == 1 else (v_new & ~(1 << bit))

                ir_u8[tensor_index] = v_new
                ir_after = ir_u8.view(torch.int8)

                # ricostruisci quantized tensor e scrivi nel layer
                q_new = torch._make_per_tensor_quantized_tensor(ir_after, scale, zero_point)
                self._set_qweight_and_bias(layer, q_new, bias)

                if DEBUG_FI:
                    # rileggo dal layer per confermare che ciò che ho scritto è quello che il layer vede
                    wq_chk, _ = self._get_qweight_and_bias(layer)
                    ir_chk = wq_chk.int_repr()
                    deq_after = wq_chk.dequantize()[tensor_index].item()
                    v_after_i8 = int(ir_chk[tensor_index].item())
                    v_after_u8 = (v_after_i8 + 256) % 256

                    print(f"[FI] {layer_name}{tensor_index} bits={bits}")
                    print(f"     scale={scale:.6g} zp={zero_point}")
                    print(f"     INT8 before={v_before_i8:4d} (u8={v_before_u8:3d} bin={format(v_before_u8,'08b')})")
                    print(f"     INT8 after ={v_after_i8:4d} (u8={v_after_u8:3d} bin={format(v_after_u8,'08b')})")
                    print(f"     DEQ  before={deq_before:.7f}  after={deq_after:.7f}\n")

        except Exception as e:
            print(f"ERROR modifying {layer_name}, index {tensor_index}: {e}")

    def restore_golden(self):
        if not self.golden_values:
            print(" No golden values stored, skipping restore.")
            return

        with torch.no_grad():
            for (layer_name, tensor_index), golden_int in self.golden_values.items():
                try:
                    layer = dict(self.network.named_modules())[layer_name]
                    wq, bias = self._get_qweight_and_bias(layer)
                    if wq.qscheme() != torch.per_tensor_affine:
                        raise NotImplementedError("Gestito solo per per-tensor.")

                    scale = wq.q_scale()
                    zero_point = wq.q_zero_point()

                    ir = wq.int_repr().clone()
                    ir[tensor_index] = np.int8(golden_int).item()   # ripristino

                    q_rest = torch._make_per_tensor_quantized_tensor(ir, scale, zero_point)
                    self._set_qweight_and_bias(layer, q_rest, bias)

                    if DEBUG_FI:
                        wq_chk, _ = self._get_qweight_and_bias(layer)
                        v_chk = int(wq_chk.int_repr()[tensor_index].item())
                        ok = (v_chk == np.int8(golden_int).item())
                        print(f"[RESTORE] {layer_name}{tensor_index} -> {v_chk} (ok={ok})")

                except Exception as e:
                    print(f"ERROR restoring {layer_name}, index {tensor_index}: {e}")

        self.golden_values.clear()

