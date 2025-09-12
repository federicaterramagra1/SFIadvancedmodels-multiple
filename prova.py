#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Campagna "online minimal" ma limitata a un sottoinsieme di bit-plane (es. bit 6 e 7).
- ESAUSTIVA forzata per N <= 3
- STATISTICA (random) oltre
- Output separato con suffisso "bits67" (o quello scelto)
"""

import os
import time
import math
import random
import heapq
from itertools import islice, product, combinations

import torch
from tqdm import tqdm

from faultManager.WeightFault import WeightFault
from faultManager.WeightFaultInjector import WeightFaultInjector
from utils import get_loader, load_from_dict, get_network
import SETTINGS


# ============================= Config locali (puoi anche metterle in SETTINGS) =============================

# Filtra i bit-plane da considerare. Default: alti.
BITS_FILTER = getattr(SETTINGS, "BITS_FILTER", [6, 7])

# Forza esaustiva fino a N <= 3
EXHAUSTIVE_UP_TO_N = getattr(SETTINGS, "EXHAUSTIVE_UP_TO_N", 3)

# Quante injection in modalità statistica se non esaustiva
MAX_FAULTS_STAT = getattr(SETTINGS, "MAX_FAULTS", 1000)

# Seed random (solo per modalità statistica)
SEED = getattr(SETTINGS, "SEED", 0)


# ============================= Helper quantizzazione & fault =============================

def _get_quant_weight(module):
    """Tensore dei pesi quantizzati (compat diverse PyTorch)."""
    if hasattr(module, "weight"):
        try:
            return module.weight()
        except Exception:
            pass
    if hasattr(module, "_packed_params") and hasattr(module._packed_params, "_weight_bias"):
        w, _ = module._packed_params._weight_bias()
        return w
    raise RuntimeError("Impossibile ottenere il peso quantizzato dal modulo.")


def _build_all_faults(model, bit_filter=None, as_list=True):
    """
    Enumera fault elementari (layer_name, tensor_index, bit) sui moduli quantizzati.
    - bit_filter: lista/insieme di bit ammessi (es. {6,7}); None => tutti 0..7.
    - as_list: True => materializza lista (ok per MLP piccoli/medi); False => generatore.
    """
    bit_ok = set(bit_filter) if bit_filter is not None else None

    def _iter():
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.quantized.Linear, torch.nn.quantized.Conv2d)):
                try:
                    w = _get_quant_weight(module)
                except Exception:
                    continue
                shape = w.shape
                for idx in product(*[range(s) for s in shape]):
                    for bit in range(8):
                        if bit_ok is not None and bit not in bit_ok:
                            continue
                        yield (name, idx, bit)

    return list(_iter()) if as_list else _iter()


def random_combination_generator(pool, r, seed=None, max_yield=None):
    """
    Genera combinazioni casuali (senza ripetizione all'interno della combo).
    Niente 'seen' globale: possibili ripetizioni tra injection diverse (va bene per stime).
    """
    rnd = random.Random(seed)
    n = len(pool)
    if n == 0:
        return
        yield  # no-op
    r = min(r, n)
    produced = 0
    while max_yield is None or produced < max_yield:
        idxs = rnd.sample(range(n), r)
        yield tuple(pool[i] for i in idxs)
        produced += 1


# ============================= Build, quantize, clean pass =============================

def build_and_quantize_once():
    # Solo CPU (x86): quantization engine
    torch.backends.quantized.engine = "fbgemm"
    device = torch.device("cpu")

    # Modello
    model = get_network(SETTINGS.NETWORK_NAME, device, SETTINGS.DATASET_NAME)
    model.to(device).eval()

    # Checkpoint se esiste
    ckpt = f"./trained_models/{SETTINGS.DATASET_NAME}_{SETTINGS.NETWORK_NAME}_trained.pth"
    if os.path.exists(ckpt):
        load_from_dict(model, device, ckpt)
        print("✓ modello caricato")
    else:
        print(f"⚠️ checkpoint non trovato: {ckpt} (proseguo senza)")

    # Loader (train per calibrazione; test per valutazione)
    train_loader, _, test_loader = get_loader(
        dataset_name=SETTINGS.DATASET_NAME,
        batch_size=SETTINGS.BATCH_SIZE,
        network_name=SETTINGS.NETWORK_NAME
    )

    # Quantize se supportato
    if hasattr(model, "quantize_model"):
        model.to(device)
        model.quantize_model(calib_loader=train_loader)
        model.eval()
        print("✓ quantizzazione completata")
    else:
        print("⚠️ modello non quantizzabile (salto quantizzazione)")

    # Clean predictions per batch
    clean_by_batch = []
    with torch.inference_mode():
        for xb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            clean_by_batch.append(torch.argmax(logits, dim=1).cpu())

    # Micro-check: 3 fault nel sottoinsieme filtrato
    injector = WeightFaultInjector(model)
    preview = list(islice(_build_all_faults(model, bit_filter=BITS_FILTER, as_list=False), 3))
    if len(preview) == 0:
        raise RuntimeError(f"Nessun sito trovato con il filtro bit={BITS_FILTER}.")
    xb_check, _ = next(iter(test_loader))
    xb_check = xb_check.to(device)
    with torch.inference_mode():
        clean_logits = model(xb_check).detach().cpu()
    for (lname, idx, bit) in preview:
        faults = [WeightFault(injection=0, layer_name=lname, tensor_index=idx, bits=[bit])]
        try:
            injector.inject_faults(faults, 'bit-flip')
            with torch.inference_mode():
                faulty_logits = model(xb_check).detach().cpu()
            max_delta = (faulty_logits - clean_logits).abs().max().item()
            print(f"[IMPACT] {lname}{idx} bit{bit}  max|Δ|={max_delta:.3e}")
        finally:
            injector.restore_golden()

    # Conta siti dopo filtro
    all_faults = _build_all_faults(model, bit_filter=BITS_FILTER, as_list=True)
    print(f"Siti single-bit (filtrati ai bit {sorted(BITS_FILTER)}): {len(all_faults)}")
    return model, device, test_loader, clean_by_batch


# ============================= Campaign (esaustiva fino a N<=3) =============================

def run_fault_injection(
    model,
    device,
    test_loader,
    clean_by_batch,
    N,
    bits_filter=BITS_FILTER,
    exhaustive_up_to_n=EXHAUSTIVE_UP_TO_N,
    max_faults_stat=MAX_FAULTS_STAT,
    seed=SEED,
    save_dir="results_minimal"
):
    t0 = time.time()

    dataset = SETTINGS.DATASET_NAME
    net_name = SETTINGS.NETWORK_NAME
    bs = getattr(test_loader, "batch_size", None) or SETTINGS.BATCH_SIZE
    bits_tag = "".join(str(b) for b in sorted(bits_filter))
    save_path = os.path.join(save_dir, dataset, net_name, f"batch_{bs}", f"minimal_bits{bits_tag}")
    os.makedirs(save_path, exist_ok=True)

    # Lista siti filtrati
    all_faults = _build_all_faults(model, bit_filter=bits_filter, as_list=True)
    num_faults = len(all_faults)
    if num_faults == 0:
        raise RuntimeError(f"Nessun sito con filtro bit={bits_filter}")

    if N > num_faults:
        print(f"[WARN] N={N} > num_faults={num_faults}. Ridimensiono N a {num_faults}.")
        N = num_faults

    force_exhaustive = (N <= exhaustive_up_to_n)
    total_possible = math.comb(num_faults, N)

    if force_exhaustive:
        print(f"[INFO] ESAUSTIVA forzata (N={N} ≤ {exhaustive_up_to_n}) — combinazioni={total_possible}")
        fault_combos = combinations(all_faults, N)
        max_iters = total_possible
    else:
        if total_possible <= max_faults_stat:
            print(f"[INFO] ESAUSTIVA (N={N}, combinazioni={total_possible})...")
            fault_combos = combinations(all_faults, N)
            max_iters = total_possible
        else:
            print(f"[INFO] STATISTICA: {max_faults_stat} random injection su ~C({num_faults},{N})")
            fault_combos = random_combination_generator(all_faults, r=N, seed=seed, max_yield=max_faults_stat)
            max_iters = max_faults_stat

    injector = WeightFaultInjector(model)

    TOP_K = 100
    top_heap = []  # (fr, inj_id, faults)
    sum_fr, n_injected = 0.0, 0
    total_samples = sum(len(t) for t in clean_by_batch)
    if total_samples == 0:
        raise RuntimeError("Test loader senza campioni.")

    pbar = tqdm(fault_combos, total=max_iters, desc=f"N={N} bits={bits_tag}")
    for inj_id, combo in enumerate(pbar, 1):
        faults = [WeightFault(injection=inj_id, layer_name=ln, tensor_index=ti, bits=[bt]) for (ln, ti, bt) in combo]
        try:
            injector.inject_faults(faults, 'bit-flip')

            mismatches = 0
            with torch.inference_mode():
                for (batch_i, (xb, _)) in enumerate(test_loader):
                    xb = xb.to(device)
                    logits_f = model(xb)
                    pred_f = torch.argmax(logits_f, dim=1).cpu()
                    mismatches += (pred_f != clean_by_batch[batch_i]).sum().item()

            fr = mismatches / float(total_samples)
            sum_fr += fr
            n_injected += 1

            if len(top_heap) < TOP_K:
                heapq.heappush(top_heap, (fr, inj_id, faults))
            else:
                if fr > top_heap[0][0]:
                    heapq.heapreplace(top_heap, (fr, inj_id, faults))

        finally:
            injector.restore_golden()

    avg_fr = (sum_fr / n_injected) if n_injected else 0.0

    prefix = f"{dataset}_{net_name}_minimal_N{N}_batch{bs}_bits{bits_tag}"
    output_file = os.path.join(save_path, f"{prefix}.txt")
    top_sorted = sorted(top_heap, key=lambda x: x[0], reverse=True)
    with open(output_file, "w") as f:
        f.write(f"Top-{min(TOP_K, len(top_sorted))} worst injections  (N={N}, bits={bits_tag})\n")
        f.write(f"Average FR: {avg_fr:.8f}  on {n_injected} injections\n\n")
        for rank, (fr, inj_id, faults) in enumerate(top_sorted, 1):
            desc = ", ".join(f"{ft.layer_name}[{ft.tensor_index}] bit{ft.bits[0]}" for ft in faults)
            f.write(f"{rank:3d}) Inj {inj_id:6d} | FR={fr:.6f} | {desc}\n")

    print(f"✓ salvato {output_file} – {(time.time()-t0)/60:.2f} min (avg FR={avg_fr:.6f}, injections={n_injected})")


# ============================= Main =============================

if __name__ == "__main__":
    model, device, test_loader, clean_by_batch = build_and_quantize_once()

    # Esempio: sweep su N con esaustiva fino a 3
    for N in [1, 2, 3]:
        run_fault_injection(
            model=model,
            device=device,
            test_loader=test_loader,
            clean_by_batch=clean_by_batch,
            N=N,
            bits_filter=BITS_FILTER,
            exhaustive_up_to_n=EXHAUSTIVE_UP_TO_N,
            max_faults_stat=MAX_FAULTS_STAT,
            seed=SEED,
            save_dir="results_minimal"
        )
