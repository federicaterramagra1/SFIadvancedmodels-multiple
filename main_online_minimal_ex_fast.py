#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exhaustive (combinazioni esatte) – versione *coerente* con Wilson (W) ed EP.
Obiettivo: poter confrontare a posteriori gli FR e i riepiloghi salvati.

Allineamenti chiave:
- Stessa build/quantize/clean pass di W/EP (CPU, fbgemm, clean_by_batch).
- Stessa enumerazione dei *siti* di fault quantizzati (layer, tensor_index, bit).
- Stessa metrica: FRcrit = mean_{test}( 1[pred_faulty != pred_clean] ).
- Stesse metriche di bias per injection e *global summary* (delta_max, KL, TV, entropia, agree, ecc.).
- Stesso formato di output (cartelle e .txt) + CSV con FR per injection per confronti diretti.

Output principale:
  results_minimal/{DATASET}/{NETWORK}/batch_{BATCH}/exhaustive_samepop/
    - {DATASET}_{NETWORK}_EXACTCOMB_K{K}_batch{B}.txt       (FR avg, Top-100, summary globale)
    - {DATASET}_{NETWORK}_EXACTCOMB_K{K}_perinj.csv         (una riga per injection: FR e bias)
    - {DATASET}_{NETWORK}_EXACTCOMB_K{K}_FR.txt             (solo riga 'FRcrit_avg=...')

Uso tipico:
  # Default: esegue K=1 e poi K=2
  python main_online_minimal_ex_fast.py

  # Solo K specifici
  python main_online_minimal_ex_fast.py --K 2
  python main_online_minimal_ex_fast.py --K 1 2

  # Sharding su 8 parti (applicato a ciascun K)
  python main_online_minimal_ex_fast.py --K 3 --shard 1 --shards 8

  # Early stop (debug) – applicato a ciascun K
  python main_online_minimal_ex_fast.py --K 3 --limit 2000000
"""

import os
import math
import csv
import argparse
import heapq
from itertools import product, combinations
from typing import List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm

import SETTINGS
from utils import get_loader, load_from_dict, get_network, load_quantized_model, save_quantized_model

from faultManager.WeightFault import WeightFault
from faultManager.WeightFaultInjector import WeightFaultInjector


# ============================= Utils comuni (allineati a W/EP) =============================

def _sci_format_comb(n: int, k: int) -> str:
    if k < 0 or k > n:
        return "0"
    k = min(k, n - k)
    if k == 0:
        return "1"
    ln10 = math.log(10.0)
    log10_val = (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / ln10
    exp = int(math.floor(log10_val))
    mant = 10 ** (log10_val - exp)
    return f"{mant:.3f}e+{exp:d}"


def _get_quant_weight(module: torch.nn.Module) -> torch.Tensor:
    """Compatibile con diverse versioni PyTorch quantized (come W/EP)."""
    if hasattr(module, "weight"):
        try:
            return module.weight()
        except Exception:
            pass
    if hasattr(module, "_packed_params") and hasattr(module._packed_params, "_weight_bias"):
        w, _ = module._packed_params._weight_bias()
        return w
    raise RuntimeError("Impossibile ottenere il peso quantizzato dal modulo.")


def _build_all_faults(model: torch.nn.Module, as_list: bool = True):
    """Tutti i *siti* single-bit (layer_name, tensor_index, bit) – identico a W/EP."""
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
                        yield (name, idx, bit)
    return list(_iter()) if as_list else _iter()


def build_and_quantize_once():
    """Stessa procedura di W/EP ma riusando il modello quantizzato salvato quando presente."""
    torch.backends.quantized.engine = "fbgemm"
    device = torch.device("cpu")

    ds_name = getattr(SETTINGS, "DATASET_NAME", getattr(SETTINGS, "DATASET", ""))
    net_name = getattr(SETTINGS, "NETWORK_NAME", getattr(SETTINGS, "NETWORK", ""))

    # Loader (test sempre, train solo per eventuale calibrazione)
    train_loader, _, test_loader = get_loader(
        dataset_name=ds_name,
        batch_size=getattr(SETTINGS, "BATCH_SIZE", 64),
        network_name=net_name,
    )

    # 1) Tenta load del quantizzato
    qmodel, qpath = load_quantized_model(ds_name, net_name, device="cpu", engine="fbgemm")
    if qmodel is not None:
        model = qmodel
        print(f"[PTQ] Quantized model caricato: {qpath}")
    else:
        # 2) Fallback: build float + ckpt + PTQ + save
        model = get_network(net_name, device, ds_name)
        model.to(device).eval()

        ckpt = f"./trained_models/{ds_name}_{net_name}_trained.pth"
        if os.path.exists(ckpt):
            load_from_dict(model, device, ckpt)
            print("[CKPT] modello float caricato")
        else:
            print(f"[WARN] checkpoint non trovato: {ckpt} (proseguo senza)")

        if hasattr(model, "quantize_model"):
            model.quantize_model(calib_loader=train_loader)
            model.eval()
            qsave = save_quantized_model(model, ds_name, net_name, engine="fbgemm")
            print(f"[PTQ] quantizzazione completata (CPU) e salvata: {qsave}")
        else:
            print("[PTQ] modello non quantizzabile – salto")

    # Predizioni clean (post-PTQ)
    clean_by_batch: List[torch.Tensor] = []
    with torch.inference_mode():
        for xb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            clean_by_batch.append(torch.argmax(logits, dim=1).cpu())

    clean_flat = torch.cat(clean_by_batch, dim=0) if len(clean_by_batch) else torch.tensor([], dtype=torch.long)
    num_classes = int(clean_flat.max().item()) + 1 if clean_flat.numel() > 0 else 2
    baseline_hist = np.bincount(clean_flat.numpy(), minlength=num_classes) if clean_flat.numel() > 0 else np.zeros(2, dtype=int)
    baseline_dist = baseline_hist / max(1, baseline_hist.sum())
    print(f"[BASELINE] pred dist = {baseline_dist.tolist()}")

    # Conteggio siti
    all_faults = _build_all_faults(model, as_list=True)
    print(f"[SITI] single-bit totali M = {len(all_faults)}")
    if len(all_faults) == 0:
        raise RuntimeError("Nessun sito enumerato: verifica quantizzazione e layer target.")

    return model, device, test_loader, clean_by_batch, baseline_hist, baseline_dist, num_classes, all_faults

# ============================= Valutazione injection (coerente a W/EP) =============================

def _evaluate_combo(model: torch.nn.Module,
                    device: torch.device,
                    test_loader,
                    clean_by_batch: List[torch.Tensor],
                    injector: WeightFaultInjector,
                    combo: Tuple[tuple, ...],
                    inj_id: int,
                    total_samples: int,
                    baseline_hist: np.ndarray,
                    baseline_dist: np.ndarray,
                    num_classes: int):
    """
    Ritorna: (frcrit, faults, bias, fault_hist, mism_by_clean, cnt_by_clean, cm_cf)
    – stesso schema di W/EP.
    """
    faults = [WeightFault(injection=inj_id, layer_name=ln, tensor_index=ti, bits=[bt]) for (ln, ti, bt) in combo]
    try:
        injector.inject_faults(faults, 'bit-flip')

        mismatches = 0
        fault_hist = np.zeros_like(baseline_hist, dtype=np.int64)
        mism_by_clean = np.zeros(num_classes, dtype=np.int64)
        cnt_by_clean  = np.zeros(num_classes, dtype=np.int64)
        cm_cf = np.zeros((num_classes, num_classes), dtype=np.int64)

        with torch.inference_mode():
            for (batch_i, (xb, _)) in enumerate(test_loader):
                xb = xb.to(device)
                pred_f = torch.argmax(model(xb), dim=1).cpu().numpy()
                np.add.at(fault_hist, pred_f, 1)

                clean_pred = clean_by_batch[batch_i].numpy()
                mism = (pred_f != clean_pred)
                mismatches += int(mism.sum())

                # Confusione clean -> fault
                np.add.at(cm_cf, (clean_pred, pred_f), 1)

                # per-classe (BER)
                for c in range(num_classes):
                    msk = (clean_pred == c)
                    cnt_by_clean[c]  += int(msk.sum())
                    if msk.any():
                        mism_by_clean[c] += int((pred_f[msk] != c).sum())

        frcrit = mismatches / float(total_samples)

        # Bias metrics per injection
        fault_total = max(1, int(fault_hist.sum()))
        fault_dist = fault_hist / fault_total
        maj_cls = int(fault_hist.argmax()) if fault_hist.size > 0 else -1
        maj_share = float(fault_hist.max()) / fault_total if fault_hist.size > 0 else 0.0
        delta_max = float(np.max(np.abs(fault_dist - baseline_dist))) if fault_hist.size > 0 else 0.0
        eps = 1e-12
        kl = float(np.sum(fault_dist * np.log((fault_dist + eps) / (baseline_dist + eps)))) if fault_hist.size > 0 else 0.0

        # Simmetria dei flip (multi-classe)
        off_sum = int(cm_cf.sum() - np.trace(cm_cf))
        asym_num = 0
        if num_classes >= 2:
            diff = np.abs(cm_cf - cm_cf.T)
            asym_num = int(diff[np.triu_indices(num_classes, k=1)].sum())
        flip_asym = float(asym_num) / max(1, off_sum)
        agree = float(np.trace(cm_cf)) / max(1, int(cm_cf.sum()))

        bias = {
            "maj_cls": maj_cls,
            "maj_share": maj_share,
            "delta_max": delta_max,
            "kl": kl,
            "flip_asym": flip_asym,
            "agree": agree,
        }

    finally:
        injector.restore_golden()

    return frcrit, faults, bias, fault_hist, mism_by_clean, cnt_by_clean, cm_cf


# ============================= Exhaustive campaign (exact combinations) =============================

def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def run_exhaustive_campaign(model: torch.nn.Module,
                            device: torch.device,
                            test_loader,
                            clean_by_batch: List[torch.Tensor],
                            baseline_hist: np.ndarray,
                            baseline_dist: np.ndarray,
                            num_classes: int,
                            all_faults: List[tuple],
                            K: int,
                            save_dir: str,
                            dataset_name: str,
                            net_name: str,
                            bs: int,
                            shard: int = 1,
                            shards: int = 1,
                            limit: Optional[int] = None,
                            top_k: int = 100):
    """
    Enumera tutte (o una sottopopolazione) delle combinazioni esatte di K fault *senza duplicati*.
    Sharding via modulo su indice di enumerazione (i % shards == shard-1).
    """
    os.makedirs(save_dir, exist_ok=True)

    m = len(all_faults)
    if not (1 <= K <= m):
        raise ValueError(f"K fuori range: K={K}, M={m}")

    N_pop = math.comb(m, K)
    print(f"[EXA] M={m}  K={K}  C(M,K)≈{_sci_format_comb(m,K)} (esatto {N_pop})")

    total_samples = sum(len(t) for t in clean_by_batch)

    injector = WeightFaultInjector(model)

    # Aggregatori globali
    global_fault_hist = np.zeros_like(baseline_hist, dtype=np.int64)
    mism_by_clean_sum = np.zeros(num_classes, dtype=np.int64)
    cnt_by_clean_sum  = np.zeros(num_classes, dtype=np.int64)
    cm_cf_sum = np.zeros((num_classes, num_classes), dtype=np.int64)

    # Top-K per FRcrit
    top_heap: List[Tuple[float, int, List[WeightFault], dict]] = []  # (fr, -inj_id, faults, bias)

    # CSV per injection
    perinj_rows: List[List] = []

    # Pre-calcolo grandezza shard (solo per tqdm)
    combos_this_shard = 0
    for i, _ in enumerate(combinations(range(m), K)):
        if (i % shards) == (shard - 1):
            combos_this_shard += 1

    print(f"[EXA] shard {shard}/{shards} – combos in shard: {combos_this_shard}")
    pbar_total = combos_this_shard if (limit is None or limit > combos_this_shard) else limit

    processed = 0
    inj_counter = 0
    with tqdm(total=pbar_total, desc=f"Exhaustive K={K} | shard {shard}/{shards}", mininterval=0.5) as pbar:
        for i, idxs in enumerate(combinations(range(m), K)):
            if (i % shards) != (shard - 1):
                continue
            if limit is not None and processed >= limit:
                break

            inj_counter += 1
            combo = tuple(all_faults[j] for j in idxs)
            frcrit, faults, bias, fault_hist, mism_by_clean, cnt_by_clean, cm_cf = _evaluate_combo(
                model, device, test_loader, clean_by_batch,
                injector, combo, inj_counter,
                total_samples, baseline_hist, baseline_dist, num_classes
            )

            # Agg globali
            global_fault_hist += fault_hist
            mism_by_clean_sum += mism_by_clean
            cnt_by_clean_sum  += cnt_by_clean
            cm_cf_sum += cm_cf

            # Top-K
            item = (frcrit, -inj_counter, faults, bias)
            if len(top_heap) < top_k:
                heapq.heappush(top_heap, item)
            else:
                if item[0] > top_heap[0][0]:
                    heapq.heapreplace(top_heap, item)

            # CSV riga per injection
            perinj_rows.append([
                inj_counter,
                frcrit,
                bias.get("maj_cls", -1),
                bias.get("maj_share", 0.0),
                bias.get("delta_max", 0.0),
                bias.get("kl", 0.0),
                bias.get("flip_asym", 0.0),
                bias.get("agree", 0.0),
                # faults compact
                "|".join([f"{f.layer_name}:{tuple(f.tensor_index)}:b{f.bits[0]}" for f in faults]),
            ])

            processed += 1
            pbar.update(1)

    # Media FRcrit sulla popolazione enumerata
    fr_mean = float(np.mean([row[1] for row in perinj_rows])) if perinj_rows else 0.0

    # Riepilogo globale su predizioni faulty
    ftot = int(global_fault_hist.sum())
    fault_dist_global = global_fault_hist / max(1, ftot)
    dmax = float(np.max(np.abs(fault_dist_global - baseline_dist))) if ftot else 0.0
    tv   = 0.5 * float(np.abs(fault_dist_global - baseline_dist).sum()) if ftot else 0.0
    eps = 1e-12
    klg  = float(np.sum(fault_dist_global * np.log((fault_dist_global + eps) / (baseline_dist + eps)))) if ftot else 0.0
    H_b  = _entropy(baseline_dist)
    H_g  = _entropy(fault_dist_global)

    # agree globale (diag/total) dalla matrice cm_cf_sum
    agree_g = float(np.trace(cm_cf_sum)) / max(1, int(cm_cf_sum.sum()))

    # Salvataggi
    out_dir = save_dir
    out_txt = os.path.join(out_dir, f"{dataset_name}_{net_name}_EXACTCOMB_K{K}_batch{bs}.txt")
    out_csv = os.path.join(out_dir, f"{dataset_name}_{net_name}_EXACTCOMB_K{K}_perinj.csv")
    out_fr  = os.path.join(out_dir, f"{dataset_name}_{net_name}_EXACTCOMB_K{K}_FR.txt")

    # CSV per-injection
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["inj_id", "frcrit", "maj_cls", "maj_share", "delta_max", "kl", "flip_asym", "agree", "faults"])
        for row in perinj_rows:
            w.writerow(row)

    # Top-100 (ordina desc)
    top_sorted = sorted(top_heap, key=lambda t: (t[0], t[1]), reverse=True)

    # TXT principale
    with open(out_txt, "w", encoding="utf-8") as fh:
        fh.write(f"Exhaustive exact combinations (same-population)\n")
        fh.write(f"Dataset/Net: {dataset_name}/{net_name}  |  batch={bs}\n")
        fh.write(f"M={m}  K={K}  N=C(M,K)={N_pop} ({_sci_format_comb(m,K)})\n")
        fh.write(f"Shard {shard}/{shards}  processed={processed}\n\n")

        fh.write(f"Top-{len(top_sorted)} worst injections  (N= {processed} nominal)\n")
        for rank, (fr, neg_id, faults, bias) in enumerate(top_sorted, 1):
            inj_id = -neg_id
            faults_str = "; ".join([f"{f.layer_name}{tuple(f.tensor_index)}:b{f.bits[0]}" for f in faults])
            fh.write(f"#{rank:03d}  inj={inj_id:6d}  FR={fr:.6f}  bias={bias}  |  {faults_str}\n")
        fh.write("\n")

        fh.write(f"Failure Rate (critical) medio: {fr_mean:.8f}  su {processed} gruppi\n")
        fh.write(f"baseline_pred_dist={baseline_dist.tolist()}\n")
        fh.write(
            f"global_summary_over_injections: fault_pred_dist={fault_dist_global.tolist()} "
            f"Δmax={dmax:.3f} KL={klg:.3f} TV={tv:.3f} H_baseline={H_b:.3f} H_global={H_g:.3f} ΔH={(H_g-H_b):+.3f}\n"
        )
        fh.write(f"agree_global={agree_g:.6f}\n")

    # FR-only (per confronto rapido nei tuoi script di plotting/tabella)
    with open(out_fr, "w", encoding="utf-8") as fh:
        fh.write(f"FRcrit_avg={fr_mean:.8f}\n")

    print(f"[EXA] salvato {out_txt}")
    print(f"[EXA] salvato {out_csv}")
    print(f"[EXA] FR medio = {fr_mean:.6f}   (anche in {out_fr})")


# ============================= Main =============================

def main():
    parser = argparse.ArgumentParser(description="Exhaustive exact-combination FI – coerente con W/EP")
    parser.add_argument("--K", type=int, nargs='+', default=[1, 2],
                        help="uno o più valori di K (default: 1 2)")
    parser.add_argument("--shard", type=int, default=1, help="indice shard (1-based)")
    parser.add_argument("--shards", type=int, default=1, help="# shard totali")
    parser.add_argument("--limit", type=int, default=None, help="limita #combo processate in questo shard (debug)")
    parser.add_argument("--top_k", type=int, default=100, help="quante injection salvare come Top-K")
    args = parser.parse_args()

    model, device, test_loader, clean_by_batch, baseline_hist, baseline_dist, num_classes, all_faults = build_and_quantize_once()

    ds_name = getattr(SETTINGS, "DATASET_NAME", getattr(SETTINGS, "DATASET", ""))
    net_name = getattr(SETTINGS, "NETWORK_NAME", getattr(SETTINGS, "NETWORK", ""))
    bs = getattr(SETTINGS, "BATCH_SIZE", 64)

    save_dir = f"results_minimal/{ds_name}/{net_name}/batch_{bs}/exhaustive_samepop"

    for K in sorted(set(args.K)):
        print(f"\n===== START K={K} =====")
        run_exhaustive_campaign(
            model=model,
            device=device,
            test_loader=test_loader,
            clean_by_batch=clean_by_batch,
            baseline_hist=baseline_hist,
            baseline_dist=baseline_dist,
            num_classes=num_classes,
            all_faults=all_faults,
            K=K,
            save_dir=save_dir,
            dataset_name=ds_name,
            net_name=net_name,
            bs=bs,
            shard=args.shard,
            shards=args.shards,
            limit=args.limit,
            top_k=args.top_k,
        )
        print(f"===== END   K={K} =====\n")


if __name__ == "__main__":
    main()
