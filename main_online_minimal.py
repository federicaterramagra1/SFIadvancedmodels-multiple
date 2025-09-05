#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import time
import os
import random
from itertools import islice, product, combinations
from tqdm import tqdm
import heapq
import math

from faultManager.WeightFault import WeightFault
from faultManager.WeightFaultInjector import WeightFaultInjector
from utils import get_loader, load_from_dict, get_network
import SETTINGS


def _get_quant_weight(module):
    """API-compat: preferisci module.weight(), fallback a _packed_params._weight_bias()."""
    # API pubblica
    if hasattr(module, "weight"):
        try:
            return module.weight()
        except Exception:
            pass
    # Fallback
    if hasattr(module, "_packed_params") and hasattr(module._packed_params, "_weight_bias"):
        w, _ = module._packed_params._weight_bias()
        return w
    raise RuntimeError("Impossibile ottenere il peso quantizzato dal modulo.")


def _build_all_faults(model):
    faults = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.quantized.Linear, torch.nn.quantized.Conv2d)):
            try:
                w = _get_quant_weight(module)
            except Exception:
                continue
            shape = w.shape
            for idx in product(*[range(s) for s in shape]):
                for bit in range(8):
                    faults.append((name, idx, bit))
    return faults


def random_combination_generator(pool, r, seed=None):
    if seed is not None:
        random.seed(seed)
    seen = set()
    while True:
        combo = tuple(sorted(random.sample(pool, r)))
        if combo not in seen:
            seen.add(combo)
            yield combo


def build_and_quantize_once():
    device = torch.device("cpu")
    model = get_network(SETTINGS.NETWORK_NAME, device, SETTINGS.DATASET_NAME)
    model.eval()

    ckpt = f"./trained_models/{SETTINGS.DATASET_NAME}_{SETTINGS.NETWORK_NAME}_trained.pth"
    load_from_dict(model, device, ckpt)
    print("✓ modello caricato")

    train_loader, _, _ = get_loader(
        dataset_name=SETTINGS.DATASET_NAME,
        batch_size=SETTINGS.BATCH_SIZE,
        network_name=SETTINGS.NETWORK_NAME
    )

    if hasattr(model, "quantize_model"):
        model.quantize_model(train_loader)
        model.eval()  # assicurati eval anche dopo la quantizzazione
        print("✓ quantizzazione completata")

    _, _, test_loader = get_loader(
        network_name=SETTINGS.NETWORK_NAME,
        batch_size=SETTINGS.BATCH_SIZE,
        dataset_name=SETTINGS.DATASET_NAME
    )

    clean_preds = []
    with torch.no_grad():
        for data, _ in test_loader:
            clean_preds.append(torch.argmax(model(data.to(device)), 1).cpu())
    clean_preds = torch.cat(clean_preds)

    # --- MICRO-CHECK di impatto fault ---
    injector = WeightFaultInjector(model)
    xb, yb = next(iter(test_loader))
    with torch.no_grad():
        clean_logits = model(xb.to(device)).cpu()

    for (lname, idx, bit) in islice(_build_all_faults(model), 3):  # prova 3 fault
        faults = [WeightFault(injection=0, layer_name=lname, tensor_index=idx, bits=[bit])]
        injector.inject_faults(faults, 'bit-flip')
        with torch.no_grad():
            faulty_logits = model(xb.to(device)).cpu()
        delta = (faulty_logits - clean_logits).abs().max().item()
        print(f"[IMPACT] {lname}{idx} bit{bit}  max|Δ|={delta:.3e}")
        injector.restore_golden()
    # ------------------------------------


    return model, test_loader, clean_preds


def run_fault_injection(model, test_loader, clean_preds, N, MAX_FAULTS=4_000_000, save_dir="results_summary"):
    t0 = time.time()
    dataset = SETTINGS.DATASET_NAME
    net_name = SETTINGS.NETWORK_NAME
    bs = SETTINGS.BATCH_SIZE
    device = torch.device("cpu")
    save_path = os.path.join(save_dir, dataset, net_name, f"batch_{bs}", "minimal")
    os.makedirs(save_path, exist_ok=True)

    all_faults = _build_all_faults(model)
    if len(all_faults) == 0:
        raise RuntimeError("Nessun fault enumerato: verifica che il modello sia realmente quantizzato.")

    # attenzione: math.comb può essere enorme; non materializzare in RAM
    total_possible = math.comb(len(all_faults), N) if N <= len(all_faults) else 0

    if total_possible and total_possible <= MAX_FAULTS:
        print(f"[INFO] Campagna ESAUSTIVA (N={N}, {total_possible} combinazioni)...")
        fault_combos = combinations(all_faults, N)  # generatore, NON castare a list
    else:
        print(f"[INFO] Campagna STATISTICA: eseguo {MAX_FAULTS} random injection su {total_possible:.2e} (N={N})")
        fault_combos = islice(random_combination_generator(all_faults, N, SETTINGS.SEED), MAX_FAULTS)

    injector = WeightFaultInjector(model)
    top_100 = []
    sum_fr = 0.0
    n_injected = 0

    for inj_id, combo in enumerate(tqdm(fault_combos, desc=f"N={N}"), 1):
        faults = [WeightFault(injection=inj_id, layer_name=layer, tensor_index=idx, bits=[bit]) for layer, idx, bit in combo]
        with torch.no_grad():
            injector.inject_faults(faults, 'bit-flip')
            preds = [torch.argmax(model(data.to(device)), 1).cpu() for data, _ in test_loader]
            injector.restore_golden()
        faulty_preds = torch.cat(preds)
        fr = 1 - (faulty_preds == clean_preds).sum().item() / len(clean_preds)
        sum_fr += fr
        n_injected += 1

        if len(top_100) < 100:
            heapq.heappush(top_100, (fr, inj_id, faults))
        else:
            heapq.heappushpop(top_100, (fr, inj_id, faults))

    avg_fr = sum_fr / n_injected if n_injected else 0.0
    prefix = f"{dataset}_{net_name}_minimal_N{N}_batch0"
    output_file = os.path.join(save_path, f"{prefix}_{dataset}.txt")
    with open(output_file, "w") as f:
        f.write(f"Top-100 worst injections  (N={N})\n")
        f.write(f"Average FR: {avg_fr:.8f}  on {n_injected} injections\n\n")
        for rank, (fr, inj_id, faults) in enumerate(sorted(top_100, reverse=True), 1):
            desc = ", ".join(f"{ft.layer_name}[{ft.tensor_index}] bit{ft.bits[0]}" for ft in faults)
            f.write(f"{rank:3d}) Inj {inj_id:6d} | FR={fr:.4f} | {desc}\n")
    print(f"✓ salvato {output_file} – {(time.time()-t0)/60:.2f} min")


if __name__ == "__main__":
    model, test_loader, clean_preds = build_and_quantize_once()
    for N in [1, 2, 3, 4, 5, 10, 20, 50, 100, 200]:
        run_fault_injection(model, test_loader, clean_preds, N)
