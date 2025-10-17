#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
 risultati allineati a main.py, implementazione snella e veloce.

Differenze rispetto a main.py:
- Niente salvataggi .npy / feature maps.
- Nessun manager pesante: loop FI minimale ma con la stessa fault list e la stessa preparazione rete.
- Confronto faulty vs clean fatto online per ogni batch.

Output:
results_minimal_mainlike/{DATASET}/{NETWORK}/batch_{BATCH}/minimal_mainlike/
  - {DATASET}_{NETWORK}_MAINLIKE_N{N}_batch{B}.txt  (FR medio, Top-K, riepilogo globale)

Note: usa SETTINGS per dataset/rete/ptq, fault list (CSV) e NUM_FAULTS_TO_INJECT.
"""

import os
import time
import heapq
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm

import SETTINGS
from utils import get_network, get_loader, load_from_dict
from faultManager.FaultListManager import FLManager
from faultManager.WeightFaultInjector import WeightFaultInjector
from faultManager.WeightFault import WeightFault


torch.backends.quantized.engine = "fbgemm"


# ========================== Build, PTQ e golden-by-batch (come main.py) ==========================

def _infer_device_from_model(model: torch.nn.Module) -> torch.device:
    try:
        p = next(model.parameters())
        return p.device
    except StopIteration:
        return torch.device("cpu")


def build_and_quantize_once_mainlike():
    """Costruisce rete, ricarica pesi float, applica PTQ su CPU e prepara i golden per batch.
    Se esiste CLEAN_OUTPUT_FOLDER/clean_output.npy lo usa (bit-match con main.py),
    altrimenti calcola le predizioni clean al volo.
    """
    # 1) Rete su CPU per coerenza con PTQ/FBGEMM
    device = torch.device("cpu")
    model = get_network(
        network_name=SETTINGS.NETWORK,
        device=device,
        dataset_name=SETTINGS.DATASET
    )
    model.to(device).eval()

    # 2) Loader iniziali
    train_loader, val_loader, test_loader = get_loader(
        network_name=SETTINGS.NETWORK,
        batch_size=SETTINGS.BATCH_SIZE,
        dataset_name=SETTINGS.DATASET
    )

    # 3) Carica ckpt (float) o allena fuori da questo script
    ckpt = f"./trained_models/{SETTINGS.DATASET}_{SETTINGS.NETWORK}_trained.pth"
    if os.path.exists(ckpt):
        print(f"[CKPT] Carico pesi float da {ckpt}")
        load_from_dict(model, device, ckpt)
    else:
        print(f"[WARN] Checkpoint non trovato: {ckpt}. Proseguo con pesi correnti.")

    # 4) PTQ su CPU (come in main.py)
    if getattr(SETTINGS, "DO_PTQ", True) and hasattr(model, "quantize_model") and not getattr(model, "_quantized_done", False):
        print("[PTQ] Quantizzazione statica 8-bit su CPU...")
        model.to("cpu").eval()
        calib_split = getattr(SETTINGS, "CALIB_SPLIT", "train")
        calib_loader = train_loader if calib_split == "train" else (val_loader or train_loader)
        maybe_new = model.quantize_model(calib_loader=calib_loader)
        if maybe_new is not None:
            model = maybe_new
        setattr(model, "_quantized_done", True)
        print("[PTQ] Completata. Modello su CPU.")
        device = torch.device("cpu")
    else:
        model.to(device).eval()

    # 5) Ricreo i loader dopo PTQ (coerente con main.py)
    train_loader, val_loader, test_loader = get_loader(
        network_name=SETTINGS.NETWORK,
        batch_size=SETTINGS.BATCH_SIZE,
        dataset_name=SETTINGS.DATASET
    )

    # 6) Golden per batch: preferisci i golden salvati da main.py, altrimenti calcola al volo
    clean_by_batch: List[torch.Tensor] = []
    baseline_hist = None

    clean_npy = os.path.join(SETTINGS.CLEAN_OUTPUT_FOLDER, "clean_output.npy")
    if os.path.exists(clean_npy):
        print(f"[BASELINE] Carico golden da {clean_npy}")
        arr = np.load(clean_npy, allow_pickle=True)
        preds_all = []
        # Supporta sia lista di batch [B_i, C] che singolo array [N, C]
        if arr.dtype == object or (arr.ndim == 1 and isinstance(arr[0], np.ndarray)):
            for batch in arr:
                yb = torch.from_numpy(batch.argmax(axis=1)).long()
                clean_by_batch.append(yb)
                preds_all.append(yb)
        else:
            # es. array unico [N, C] -> ricostruzione by-batch seguendo il loader
            idx = 0
            with torch.inference_mode():
                for xb, _ in test_loader:
                    B = xb.shape[0]
                    part = arr[idx: idx + B]
                    idx += B
                    yb = torch.from_numpy(part.argmax(axis=1)).long()
                    clean_by_batch.append(yb)
                    preds_all.append(yb)
        all_clean = torch.cat(preds_all)
        num_classes = int(all_clean.max().item() + 1) if all_clean.numel() else 2
        baseline_hist = np.bincount(all_clean.numpy(), minlength=num_classes)
    else:
        print("[BASELINE] Golden non presente su disco: calcolo al volo")
        preds_all = []
        with torch.inference_mode():
            for xb, _ in tqdm(test_loader, desc="Clean pass (by-batch)", mininterval=0.5):
                logits = model(xb.to(device))
                y = torch.argmax(logits, dim=1).cpu()
                clean_by_batch.append(y)
                preds_all.append(y)
        all_clean = torch.cat(preds_all)
        num_classes = int(all_clean.max().item() + 1) if all_clean.numel() else 2
        baseline_hist = np.bincount(all_clean.numpy(), minlength=num_classes)

    baseline_dist = baseline_hist / max(1, baseline_hist.sum())
    print(f"[BASELINE] pred dist = {baseline_dist.tolist()}")

    return model, device, test_loader, clean_by_batch, baseline_hist, baseline_dist, num_classes


# ========================== Valutazione di UN gruppo (FI minimale) ==========================

def _evaluate_group_mainlike(model: torch.nn.Module,
                             loader: torch.utils.data.DataLoader,
                             clean_by_batch: List[torch.Tensor],
                             injector: WeightFaultInjector,
                             group: List[WeightFault],
                             inj_id: int,
                             num_classes: int,
                             fault_mode: str = 'bit-flip') -> Tuple[float, dict, np.ndarray]:
    """
    Inietta il gruppo (faults simultanei), esegue l'inferenza su TUTTI i batch,
    confronta top-1 faulty vs top-1 clean_by_batch e calcola FRcrit.
    Ritorna: (frcrit, bias_info, faulty_pred_hist)
    """
    total = sum(len(t) for t in clean_by_batch)
    mism = 0
    faulty_hist = np.zeros(num_classes, dtype=np.int64)

    dev = _infer_device_from_model(model)

    try:
        injector.inject_faults(group, fault_mode=fault_mode)

        with torch.inference_mode():
            for b_id, (xb, _) in enumerate(loader):
                logits_f = model(xb.to(dev))
                yy_f = torch.argmax(logits_f, dim=1).cpu()

                yy_c = clean_by_batch[b_id]
                mism += int((yy_f != yy_c).sum().item())

                # agg. distribuzione faulty
                np.add.at(faulty_hist, yy_f.numpy(), 1)

    finally:
        injector.restore_golden()

    frcrit = mism / float(total)

    # bias essenziale per ranking
    ftot = int(faulty_hist.sum())
    if ftot > 0:
        maj_cls = int(np.argmax(faulty_hist))
        maj_share = float(faulty_hist.max()) / ftot
        agree = 1.0 - frcrit
    else:
        maj_cls, maj_share, agree = -1, 0.0, 0.0

    bias = {"maj_cls": maj_cls, "maj_share": maj_share, "agree": agree}
    return frcrit, bias, faulty_hist


# ========================== Campagna MAINLIKE (usa CSV della fault list) ==========================

def run_fault_injection_mainlike(model,
                                 device,
                                 test_loader,
                                 clean_by_batch,
                                 baseline_hist,
                                 baseline_dist,
                                 num_classes,
                                 N: int,
                                 save_dir="results_minimal_mainlike",
                                 top_k: int = 100):
    """
    Esegue la campagna usando la fault list di FLManager (come main.py), ma senza salvataggi .npy.
    Mantiene Top-K per FRcrit e scrive un riepilogo globale.
    """
    t0 = time.time()
    dataset = SETTINGS.DATASET
    net = SETTINGS.NETWORK
    bs = getattr(test_loader, "batch_size", SETTINGS.BATCH_SIZE)

    save_path = os.path.join(save_dir, dataset, net, f"batch_{bs}", "minimal_mainlike")
    os.makedirs(save_path, exist_ok=True)
    prefix = f"{dataset}_{net}_MAINLIKE_N{N}_batch{bs}"
    out_file = os.path.join(save_path, f"{prefix}.txt")

    # ---- Fault list come in main.py ----
    fl_manager = FLManager(
        network=model,
        network_name=SETTINGS.NETWORK,
        device=device,
        module_class=getattr(SETTINGS, "MODULE_CLASSES_FAULT_LIST", None)
    )
    fault_groups = fl_manager.get_weight_fault_list()  # gruppi dal CSV (Injection ID)

    cap = getattr(SETTINGS, "FAULTS_TO_INJECT", -1)
    if cap is not None and cap > 0 and cap < len(fault_groups):
        import random
        random.seed(getattr(SETTINGS, "SEED", 0))
        fault_groups = random.sample(fault_groups, cap)
        print(f"[INFO] Fault list limitata a {cap} gruppi.")
    else:
        print(f"[INFO] Fault list esaustiva: {len(fault_groups)} gruppi (ognuno da {SETTINGS.NUM_FAULTS_TO_INJECT} bit).")

    injector = WeightFaultInjector(model)
    total_samples_test = sum(len(t) for t in clean_by_batch)

    # aggregatori
    global_fault_hist = np.zeros_like(baseline_hist, dtype=np.int64)
    sum_fr = 0.0
    n_inj = 0
    top_heap = []  # (frcrit, inj_id, faults, bias)

    pbar = tqdm(enumerate(fault_groups), total=len(fault_groups), desc="FI (mainlike)", mininterval=1.0)
    for inj_id, group in pbar:
        frcrit, bias, fh = _evaluate_group_mainlike(
            model=model,
            loader=test_loader,
            clean_by_batch=clean_by_batch,
            injector=injector,
            group=group,
            inj_id=inj_id,
            num_classes=num_classes,
            fault_mode=getattr(SETTINGS, "FAULT_MODEL", "bit-flip")
        )
        sum_fr += frcrit
        n_inj += 1
        global_fault_hist += fh

        if len(top_heap) < top_k:
            heapq.heappush(top_heap, (frcrit, inj_id, group, bias))
        elif frcrit > top_heap[0][0]:
            heapq.heapreplace(top_heap, (frcrit, inj_id, group, bias))

    avg_fr = (sum_fr / n_inj) if n_inj else 0.0
    top_sorted = sorted(top_heap, key=lambda x: x[0], reverse=True)

    # riepilogo globale stile minimal
    total_preds = int(global_fault_hist.sum())
    global_fault_dist = (global_fault_hist / total_preds) if total_preds > 0 else baseline_dist
    eps = 1e-12
    delta_max = float(np.max(np.abs(global_fault_dist - baseline_dist)))
    kl = float(np.sum(global_fault_dist * np.log((global_fault_dist + eps) / (baseline_dist + eps))))
    tv = 0.5 * float(np.sum(np.abs(global_fault_dist - baseline_dist)))

    H = lambda p: float(-np.sum(p * np.log(p + eps)))
    Hb = H(baseline_dist)
    Hg = H(global_fault_dist)
    dH = Hb - Hg

    with open(out_file, "w") as f:
        f.write(f"Top-{min(top_k, len(top_sorted))} worst injections  (N={N} nominal)\n")
        f.write(f"Failure Rate (critical) medio: {avg_fr:.8f}  su {n_inj} gruppi\n")
        f.write(f"baseline_pred_dist={baseline_dist.tolist()}\n")
        f.write(
            "global_summary_over_injections: "
            f"fault_pred_dist={global_fault_dist.tolist()} "
            f"Δmax={delta_max:.3f} KL={kl:.3f} TV={tv:.3f} "
            f"H_baseline={Hb:.3f} H_global={Hg:.3f} ΔH={dH:.3f}\n\n"
        )
        for rank, (frcrit, inj_id, group, bias) in enumerate(top_sorted, 1):
            desc = ", ".join(f"{ft.layer_name}[{ft.tensor_index}] bit{ft.bits[0]}" for ft in group)
            f.write(
                f"{rank:3d}) Inj {inj_id:6d} | FRcrit={frcrit:.6f} | "
                f"maj={bias['maj_cls']}@{bias['maj_share']:.2f} "
                f"agree={bias['agree']:.3f} | {desc}\n"
            )

    dt = (time.time() - t0) / 60.0
    tot_evals_like_main = total_samples_test * n_inj
    print(
        f"[MAINLIKE_MINIMAL] salvato {out_file} – {dt:.2f} min "
        f"(avg FR={avg_fr:.6f}, injections={n_inj}, tot_evals≈{tot_evals_like_main})"
    )
    return avg_fr, n_inj, top_sorted, out_file


# =================================== Main ===================================

if __name__ == "__main__":
    # Per evitare oversubscription su CPU
    try:
        import psutil
        torch.set_num_threads(psutil.cpu_count(logical=False) or os.cpu_count() or 1)
    except Exception:
        torch.set_num_threads(os.cpu_count() or 1)

    # Build + PTQ + golden pass (by-batch), come in main.py
    model, device, test_loader, clean_by_batch, baseline_hist, baseline_dist, num_classes = build_and_quantize_once_mainlike()

    # Esecuzione (N è solo etichetta nel nome file; i gruppi arrivano dal CSV costruito con NUM_FAULTS_TO_INJECT)
    N_nominale = getattr(SETTINGS, "NUM_FAULTS_TO_INJECT", 1)
    run_fault_injection_mainlike(
        model=model,
        device=device,
        test_loader=test_loader,
        clean_by_batch=clean_by_batch,
        baseline_hist=baseline_hist,
        baseline_dist=baseline_dist,
        num_classes=num_classes,
        N=N_nominale,
        save_dir="results_minimal_mainlike",
        top_k=100
    )
