#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import random
import heapq
from itertools import islice, product, combinations
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import torch
from tqdm import tqdm

from faultManager.WeightFault import WeightFault
from faultManager.WeightFaultInjector import WeightFaultInjector
from utils import get_loader, load_from_dict, get_network
import SETTINGS


# ============================= Utils numerici & stampa sicura =============================

def _sci_format_comb(n: int, k: int) -> str:
    """Formatta C(n,k) in scientifica usando lgamma (no overflow)."""
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


# ============================= Helper quantizzazione & fault =============================

def _get_quant_weight(module):
    """Prova a prendere i pesi quantizzati in modo robusto a versioni PyTorch diverse."""
    if hasattr(module, "weight"):
        try:
            return module.weight()
        except Exception:
            pass
    if hasattr(module, "_packed_params") and hasattr(module._packed_params, "_weight_bias"):
        w, _ = module._packed_params._weight_bias()
        return w
    raise RuntimeError("Impossibile ottenere il peso quantizzato dal modulo.")


def _build_all_faults(model, as_list=True):
    """
    Enumera tutti i fault elementari (layer_name, tensor_index, bit) sui moduli quantizzati.
    as_list=True => lista (ok per modelli piccoli/medi). as_list=False => generatore.
    """
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


# ============================= Sampler combinazioni =============================

def srs_combinations(pool, r, seed=None, max_yield=None):
    """
    SRS i.i.d. sulle combinazioni:
    - ogni iter estrae r indici distinti (senza ripetizione *dentro* la combo)
    - tra iterazioni si accettano ripetizioni di combo (camp. con reinserimento sull'insieme di tutte le combo)
    - così le FR per injection sono i.i.d.
    """
    rnd = random.Random(seed)
    n = len(pool)
    if n == 0:
        return
        yield
    r = min(r, n)
    produced = 0
    while max_yield is None or produced < max_yield:
        idxs = rnd.sample(range(n), r)
        yield tuple(pool[i] for i in idxs)
        produced += 1


# ============================= Statistica: z, FPC, Wald, Wilson =============================

def _z_from_conf(conf: float) -> float:
    return {0.90: 1.645, 0.95: 1.96, 0.99: 2.576, 0.999: 3.291}.get(conf, 1.96)

def fpc_factor(N_pop: Optional[int], n: int) -> float:
    """FPC = sqrt((N_pop - n)/(N_pop - 1)); N_pop=None => 1.0"""
    if not N_pop or N_pop <= 0:
        return 1.0
    if n <= 1 or n >= N_pop:
        val = max(0.0, (N_pop - n) / max(1, N_pop - 1))
        return math.sqrt(val)
    return math.sqrt((N_pop - n) / (N_pop - 1))

def wald_halfwidth(p_hat: float, n: int, conf: float, N_pop: Optional[int] = None) -> float:
    z = _z_from_conf(conf)
    p = min(max(p_hat, 1e-12), 1.0 - 1e-12)
    hw = z * math.sqrt(p * (1.0 - p) / max(1, n))
    return hw * fpc_factor(N_pop, n)

def wald_ci(p_hat: float, n: int, conf: float, N_pop: Optional[int] = None):
    half = wald_halfwidth(p_hat, n, conf, N_pop)
    low = max(0.0, p_hat - half)
    high = min(1.0, p_hat + half)
    return low, high, half

def wilson_ci(p_hat: float, n: int, conf: float):
    """Wilson score CI per proporzioni (senza FPC; la correzione è già “internamente bayesiana”)."""
    z = _z_from_conf(conf)
    if n == 0:
        return 0.0, 1.0, 0.5
    p = min(max(p_hat, 1e-12), 1-1e-12)
    denom  = 1 + (z*z)/n
    center = (p + (z*z)/(2*n)) / denom
    half   = (z*math.sqrt(p*(1-p)/n + (z*z)/(4*n*n))) / denom
    low = max(0.0, center - half)
    high = min(1.0, center + half)
    return low, high, half

def plan_n_normal(p_hat: float, eps: float, conf: float) -> int:
    """n ≈ z^2 p(1-p) / eps^2 (stima iniziale, usata nel branch STAT come target morbido)."""
    z = _z_from_conf(conf)
    p = min(max(p_hat, 1e-4), 1-1e-4)
    return max(1, math.ceil((z*z*p*(1-p)) / (eps*eps)))

# ---- EP curve (per la pipeline EP) ----

def ep_next_error(e_goal: float, E_hat: float, p_hat: float) -> float:
    """E–P curve: se E_hat/3 <= e_goal -> e_goal; altrimenti parabola con apice a 0.5 e valore E_hat/3."""
    if E_hat <= 0.0:
        return e_goal
    thresh = E_hat / 3.0
    if thresh <= e_goal:
        return e_goal
    k = 4.0 * (thresh - e_goal)
    val = -k * (p_hat ** 2) + k * p_hat + e_goal
    return float(min(max(val, e_goal), thresh))

def plan_n_with_fpc(p_hat: float, e_target: float, conf: float, N_pop: Optional[int]) -> int:
    """Formula con/ senza FPC per pianificare n totale desiderato a un dato errore target."""
    z = _z_from_conf(conf)
    p = min(max(p_hat, 1e-6), 1 - 1e-6)
    if not N_pop or N_pop <= 0:
        n = math.ceil((z*z*p*(1-p)) / (e_target*e_target))
        return max(1, n)
    denom = 1.0 + (e_target*e_target*(N_pop - 1.0)) / (z*z*p*(1-p))
    n = N_pop / denom
    return int(min(N_pop, max(1, math.ceil(n))))


# ============================= Build, quantize, clean pass =============================

def build_and_quantize_once():
    # Forza quantizzazione CPU (fbgemm)
    torch.backends.quantized.engine = "fbgemm"
    device = torch.device("cpu")

    # Modello + checkpoint
    model = get_network(SETTINGS.NETWORK_NAME, device, SETTINGS.DATASET_NAME)
    model.to(device).eval()
    ckpt = f"./trained_models/{SETTINGS.DATASET_NAME}_{SETTINGS.NETWORK_NAME}_trained.pth"
    if os.path.exists(ckpt):
        load_from_dict(model, device, ckpt)
        print("modello caricato")
    else:
        print(f"checkpoint non trovato: {ckpt} (proseguo senza)")

    # Loader (train per calibrazione, test per inferenza)
    train_loader, _, test_loader = get_loader(
        dataset_name=SETTINGS.DATASET_NAME,
        batch_size=SETTINGS.BATCH_SIZE,
        network_name=SETTINGS.NETWORK_NAME
    )

    # Quantize se disponibile
    if hasattr(model, "quantize_model"):
        model.to(device)
        model.quantize_model(calib_loader=train_loader)
        model.eval()
        print("quantizzazione completata")
    else:
        print("modello non quantizzabile (salto quantizzazione)")

    # Clean predictions per batch
    clean_by_batch = []
    with torch.inference_mode():
        for xb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            clean_by_batch.append(torch.argmax(logits, dim=1).cpu())

    # Distribuzione baseline
    clean_flat = torch.cat(clean_by_batch, dim=0)
    num_classes = int(clean_flat.max().item()) + 1 if clean_flat.numel() > 0 else 2
    baseline_hist = np.bincount(clean_flat.numpy(), minlength=num_classes)
    baseline_dist = baseline_hist / max(1, baseline_hist.sum())
    print(f"[BASELINE] pred dist = {baseline_dist}")

    # Micro-check rapido su 3 fault
    injector = WeightFaultInjector(model)
    preview = list(islice(_build_all_faults(model, as_list=False), 3))
    if len(preview) == 0:
        raise RuntimeError("Nessun fault enumerato: verifica quantizzazione e layer target.")

    _test_iter = iter(test_loader)
    try:
        xb_check, _ = next(_test_iter)
    except StopIteration:
        raise RuntimeError("Test loader vuoto: impossibile eseguire il micro-check.")
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

    # Conteggio siti
    all_faults = _build_all_faults(model, as_list=True)
    print(f"Siti single-bit totali: {len(all_faults)}")
    if len(all_faults) == 0:
        raise RuntimeError("Nessun fault enumerato: impossibile proseguire.")

    return model, device, test_loader, clean_by_batch, baseline_hist, baseline_dist, num_classes, all_faults


# ============================= Valutazione singola combo =============================

def _evaluate_combo(model, device, test_loader, clean_by_batch, injector, combo, inj_id,
                    total_samples, baseline_hist, baseline_dist, num_classes):
    """
    Inietta una combo, valuta FRcrit + metriche (per compatibilità). Ritorna varie cose,
    ma in questo script useremo soprattutto frcrit.
    """
    faults = [WeightFault(injection=inj_id, layer_name=ln, tensor_index=ti, bits=[bt])
              for (ln, ti, bt) in combo]
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
                logits_f = model(xb)
                pred_f = torch.argmax(logits_f, dim=1).cpu().numpy()
                np.add.at(fault_hist, pred_f, 1)

                clean_pred = clean_by_batch[batch_i].numpy()
                mism = (pred_f != clean_pred)
                mismatches += int(mism.sum())

                np.add.at(cm_cf, (clean_pred, pred_f), 1)

                # per BER
                for c in range(num_classes):
                    msk = (clean_pred == c)
                    cnt_by_clean[c]  += int(msk.sum())
                    if msk.any():
                        mism_by_clean[c] += int((pred_f[msk] != c).sum())

        frcrit = mismatches / float(total_samples)

        # Bias metriche riassunte (non indispensabili per il confronto CI)
        fault_total = max(1, int(fault_hist.sum()))
        fault_dist = fault_hist / fault_total
        maj_cls = int(fault_hist.argmax()) if fault_hist.size > 0 else -1
        maj_share = float(fault_hist.max()) / fault_total if fault_hist.size > 0 else 0.0
        delta_max = float(np.max(np.abs(fault_dist - baseline_dist))) if fault_hist.size > 0 else 0.0
        eps = 1e-12
        kl = float(np.sum(fault_dist * np.log((fault_dist + eps) / (baseline_dist + eps)))) if fault_hist.size > 0 else 0.0

        off_sum = int(cm_cf.sum() - np.trace(cm_cf))
        diff = np.abs(cm_cf - cm_cf.T)
        asym_num = int(diff[np.triu_indices(num_classes, k=1)].sum())
        flip_asym = float(asym_num) / max(1, off_sum) if off_sum > 0 else 0.0
        agree = float(np.trace(cm_cf)) / max(1, int(cm_cf.sum())) if cm_cf.sum() > 0 else 0.0

        bias = {"maj_cls": maj_cls, "maj_share": maj_share, "delta_max": delta_max,
                "kl": kl, "flip_asym": flip_asym, "agree": agree}

    finally:
        injector.restore_golden()

    return frcrit, faults, bias, fault_hist, mism_by_clean, cnt_by_clean, cm_cf


# ============================= Pipeline EP (iterativa E–P) =============================

def run_ep_campaign(
    model, device, test_loader, clean_by_batch, all_faults, N,
    baseline_hist, baseline_dist, num_classes,
    pilot=None, e_goal=None, conf=None, block=None, budget_cap=None, seed=None,
    use_fpc: bool = False, N_pop: Optional[int] = None,
):
    pilot   = pilot   if pilot   is not None else getattr(SETTINGS, "STAT_PILOT", 200)
    e_goal  = e_goal  if e_goal  is not None else getattr(SETTINGS, "STAT_EPS",   0.005)
    conf    = conf    if conf    is not None else getattr(SETTINGS, "STAT_CONF",  0.95)
    block   = block   if block   is not None else getattr(SETTINGS, "STAT_BLOCK", 50)
    budget_cap = budget_cap if budget_cap is not None else getattr(SETTINGS, "STAT_BUDGET_CAP", None)
    seed    = seed    if seed    is not None else getattr(SETTINGS, "SEED", 0)

    if not use_fpc:
        N_pop = None

    injector = WeightFaultInjector(model)
    total_samples = sum(len(t) for t in clean_by_batch)

    # pilot
    gen = srs_combinations(all_faults, r=N, seed=seed,   max_yield=pilot)
    gen_more = srs_combinations(all_faults, r=N, seed=seed+1, max_yield=None)

    sum_fr, n_inj, inj_id = 0.0, 0, 0

    for combo in tqdm(gen, total=pilot, desc=f"[EP] pilot N={N}"):
        inj_id += 1
        frcrit, *_ = _evaluate_combo(
            model, device, test_loader, clean_by_batch, injector, combo, inj_id, total_samples,
            baseline_hist, baseline_dist, num_classes
        )
        sum_fr += frcrit
        n_inj  += 1

    p_hat = sum_fr / max(1, n_inj)
    E_hat = wald_halfwidth(p_hat, n_inj, conf, N_pop)

    steps = 0
    while True:
        steps += 1
        if E_hat <= e_goal:
            break
        if budget_cap and n_inj >= budget_cap:
            break

        e_next = ep_next_error(e_goal=e_goal, E_hat=E_hat, p_hat=p_hat)
        n_tot_desired = plan_n_with_fpc(p_hat=p_hat, e_target=e_next, conf=conf, N_pop=N_pop)
        add_needed = max(0, n_tot_desired - n_inj)
        if budget_cap:
            add_needed = min(add_needed, max(0, budget_cap - n_inj))

        to_do = add_needed
        pbar2 = tqdm(total=add_needed, desc=f"[EP] step#{steps} -> n={n_tot_desired} (e_next={e_next:.6g})")
        while to_do > 0:
            combo = next(gen_more)
            inj_id += 1
            frcrit, *_ = _evaluate_combo(
                model, device, test_loader, clean_by_batch, injector, combo, inj_id, total_samples,
                baseline_hist, baseline_dist, num_classes
            )
            sum_fr += frcrit
            n_inj  += 1
            to_do  -= 1
            pbar2.update(1)

            if (n_inj % block) == 0:
                p_hat = sum_fr / n_inj
                E_hat = wald_halfwidth(p_hat, n_inj, conf, N_pop)
                if E_hat <= e_goal or (budget_cap and n_inj >= budget_cap):
                    break

        p_hat = sum_fr / n_inj
        E_hat = wald_halfwidth(p_hat, n_inj, conf, N_pop)
        if E_hat <= e_goal or (budget_cap and n_inj >= budget_cap):
            break

    # CI a stop EP
    w_low, w_high, w_half = wald_ci(p_hat, n_inj, conf, N_pop)
    wl_low, wl_high, wl_half = wilson_ci(p_hat, n_inj, conf)
    return {
        "n": n_inj, "p_hat": p_hat,
        "wald":  {"half": w_half,  "low": w_low,  "high": w_high},
        "wilson":{"half": wl_half, "low": wl_low, "high": wl_high},
        "steps": steps,
    }


# ============================= Pipeline STAT (stop su Wilson) =============================

def run_stat_campaign(
    model, device, test_loader, clean_by_batch, all_faults, N,
    baseline_hist, baseline_dist, num_classes,
    pilot=None, eps=None, conf=None, block=None, budget_cap=None, seed=None
):
    pilot   = pilot   if pilot   is not None else getattr(SETTINGS, "STAT_PILOT", 200)
    eps     = eps     if eps     is not None else getattr(SETTINGS, "STAT_EPS",   0.005)
    conf    = conf    if conf    is not None else getattr(SETTINGS, "STAT_CONF",  0.95)
    block   = block   if block   is not None else getattr(SETTINGS, "STAT_BLOCK", 50)
    budget_cap = budget_cap if budget_cap is not None else getattr(SETTINGS, "STAT_BUDGET_CAP", None)
    seed    = seed    if seed    is not None else getattr(SETTINGS, "SEED", 0)

    injector = WeightFaultInjector(model)
    total_samples = sum(len(t) for t in clean_by_batch)

    # pilot
    gen = srs_combinations(all_faults, r=N, seed=seed, max_yield=pilot)
    sum_fr, n_inj, inj_id = 0.0, 0, 0
    for combo in tqdm(gen, total=pilot, desc=f"[STAT] pilot N={N}"):
        inj_id += 1
        frcrit, *_ = _evaluate_combo(
            model, device, test_loader, clean_by_batch, injector, combo, inj_id, total_samples,
            baseline_hist, baseline_dist, num_classes
        )
        sum_fr += frcrit
        n_inj  += 1

    p_hat = sum_fr / max(1, n_inj)
    n_target = plan_n_normal(p_hat, eps, conf)
    if budget_cap:
        n_target = min(n_target, budget_cap)

    # sequenziale con Wilson
    gen2 = srs_combinations(all_faults, r=N, seed=seed+1, max_yield=None)
    block_acc = 0
    for combo in tqdm(gen2, total=None, desc=f"[STAT] N={N} target~{n_target} (eps={eps}, conf={conf})"):
        inj_id += 1
        frcrit, *_ = _evaluate_combo(
            model, device, test_loader, clean_by_batch, injector, combo, inj_id, total_samples,
            baseline_hist, baseline_dist, num_classes
        )
        sum_fr += frcrit
        n_inj  += 1
        block_acc += 1

        if block_acc >= block:
            block_acc = 0
            p_hat = sum_fr / n_inj
            _, _, half = wilson_ci(p_hat, n_inj, conf)
            if half <= eps:
                break
            if budget_cap and n_inj >= budget_cap:
                break

    # CI a stop STAT
    p_hat = sum_fr / n_inj
    wl_low, wl_high, wl_half = wilson_ci(p_hat, n_inj, conf)
    w_low, w_high, w_half = wald_ci(p_hat, n_inj, conf)
    return {
        "n": n_inj, "p_hat": p_hat,
        "wilson":{"half": wl_half, "low": wl_low, "high": wl_high},
        "wald":  {"half": w_half,  "low": w_low,  "high": w_high},
    }


# ============================= Confronto “stesso stream” (chiave) =============================

def compare_same_stream(
    model, device, test_loader, clean_by_batch,
    baseline_hist, baseline_dist, num_classes,
    all_faults, N, eps=0.005, conf=0.95, block=50, seed=123, N_pop: Optional[int]=None
):
    """
    Un singolo stream SRS (stesse injection, stesso ordine).
    Ad ogni checkpoint calcola *entrambi* gli halfwidth su p_hat cumulativa.
    Registra:
      - n_stop_wald: primo n con half_wald <= eps
      - n_stop_wil : primo n con half_wilson <= eps
    Poi stampa CI Wald & Wilson a *entrambi* gli stop, sugli *stessi* campioni.
    """
    injector = WeightFaultInjector(model)
    total_samples = sum(len(t) for t in clean_by_batch)

    gen = srs_combinations(all_faults, r=N, seed=seed, max_yield=None)
    sum_fr, n_inj, inj_id = 0.0, 0, 0

    n_stop_wald = None
    n_stop_wil  = None
    p_at_stop_wald = None
    p_at_stop_wil  = None

    for combo in tqdm(gen, total=None, desc=f"[COMPARE-SAME-STREAM] N={N} eps={eps}, conf={conf}"):
        inj_id += 1
        frcrit, *_ = _evaluate_combo(
            model, device, test_loader, clean_by_batch, injector, combo, inj_id, total_samples,
            baseline_hist, baseline_dist, num_classes
        )
        sum_fr += frcrit
        n_inj  += 1

        if (n_inj % block) == 0:
            p_hat = sum_fr / n_inj

            # Wald (con FPC opzionale)
            w_half = wald_halfwidth(p_hat, n_inj, conf, N_pop)
            # Wilson
            _, _, wil_half = wilson_ci(p_hat, n_inj, conf)

            if n_stop_wald is None and w_half <= eps:
                n_stop_wald = n_inj
                p_at_stop_wald = p_hat

            if n_stop_wil is None and wil_half <= eps:
                n_stop_wil = n_inj
                p_at_stop_wil = p_hat

            if (n_stop_wald is not None) and (n_stop_wil is not None):
                break

    # Stampa comparativa a parità di campioni
    def both_ci(p_hat, n):
        wl, wh, wih = wilson_ci(p_hat, n, conf)
        w0, w1, wah = wald_ci(p_hat, n, conf, N_pop)
        return (wl, wh, wih), (w0, w1, wah)

    results = {}

    if n_stop_wald is not None:
        (wil_l, wil_h, wil_half), (wal_l, wal_h, wal_half) = both_ci(p_at_stop_wald, n_stop_wald)
        print(f"\n== WALD stop (same-stream) ==  n={n_stop_wald}  p_hat={p_at_stop_wald:.6f}")
        print(f"  Wald   half={wal_half:.6f}  [{wal_l:.6f}, {wal_h:.6f}]")
        print(f"  Wilson half={wil_half:.6f}  [{wil_l:.6f}, {wil_h:.6f}]")
        results["stop_wald"] = {
            "n": n_stop_wald, "p_hat": p_at_stop_wald,
            "wald":  {"half": wal_half,  "low": wal_l,  "high": wal_h},
            "wilson":{"half": wil_half, "low": wil_l, "high": wil_h},
        }

    if n_stop_wil is not None:
        (wil_l, wil_h, wil_half), (wal_l, wal_h, wal_half) = both_ci(p_at_stop_wil, n_stop_wil)
        print(f"\n== WILSON stop (same-stream) ==  n={n_stop_wil}  p_hat={p_at_stop_wil:.6f}")
        print(f"  Wilson half={wil_half:.6f}  [{wil_l:.6f}, {wil_h:.6f}]")
        print(f"  Wald   half={wal_half:.6f}  [{wal_l:.6f}, {wal_h:.6f}]")
        results["stop_wilson"] = {
            "n": n_stop_wil, "p_hat": p_at_stop_wil,
            "wilson":{"half": wil_half, "low": wil_l, "high": wil_h},
            "wald":  {"half": wal_half,  "low": wal_l,  "high": wal_h},
        }

    return results


# ============================= MAIN =============================

if __name__ == "__main__":
    # Build + quantize + clean
    (model, device, test_loader, clean_by_batch,
     baseline_hist, baseline_dist, num_classes, all_faults) = build_and_quantize_once()

    dataset = SETTINGS.DATASET_NAME
    net     = SETTINGS.NETWORK_NAME
    bs      = getattr(test_loader, "batch_size", None) or SETTINGS.BATCH_SIZE

    # Parametri statistici
    CONF = getattr(SETTINGS, "STAT_CONF", 0.95)
    EPS  = getattr(SETTINGS, "STAT_EPS",  0.005)
    BLOCK= getattr(SETTINGS, "STAT_BLOCK", 50)
    SEED = getattr(SETTINGS, "SEED", 0)

    # Lista di N da testare
    Ns = [1, 2, 3, 4, 5, 10, 50, 100]

    # Cartelle per salvare report di confronto
    base_dir = os.path.join("results_minimal", dataset, net, f"batch_{bs}", "compare_unified")
    os.makedirs(base_dir, exist_ok=True)

    for N in Ns:
        print(f"\n=== N={N} – Confronto EP, STAT e SAME-STREAM ===")

        # 1) EP (iterativo E–P con stop su Wald)
        ep_res = run_ep_campaign(
            model, device, test_loader, clean_by_batch, all_faults, N,
            baseline_hist, baseline_dist, num_classes,
            pilot=getattr(SETTINGS, "STAT_PILOT", 200),
            e_goal=EPS, conf=CONF, block=BLOCK, budget_cap=getattr(SETTINGS, "STAT_BUDGET_CAP", None),
            seed=SEED, use_fpc=False, N_pop=None
        )
        print("\n-- EP stop --")
        print(f"n_ep={ep_res['n']}  p_hat={ep_res['p_hat']:.6f}")
        print(f"  Wald   half={ep_res['wald']['half']:.6f}  [ {ep_res['wald']['low']:.6f}, {ep_res['wald']['high']:.6f} ]")
        print(f"  Wilson half={ep_res['wilson']['half']:.6f}  [ {ep_res['wilson']['low']:.6f}, {ep_res['wilson']['high']:.6f} ]")

        # 2) STAT (sequenziale, stop su Wilson)
        st_res = run_stat_campaign(
            model, device, test_loader, clean_by_batch, all_faults, N,
            baseline_hist, baseline_dist, num_classes,
            pilot=getattr(SETTINGS, "STAT_PILOT", 200),
            eps=EPS, conf=CONF, block=BLOCK, budget_cap=getattr(SETTINGS, "STAT_BUDGET_CAP", None),
            seed=SEED
        )
        print("\n-- STAT stop --")
        print(f"n_st={st_res['n']}  p_hat={st_res['p_hat']:.6f}")
        print(f"  Wilson half={st_res['wilson']['half']:.6f}  [ {st_res['wilson']['low']:.6f}, {st_res['wilson']['high']:.6f} ]")
        print(f"  Wald   half={st_res['wald']['half']:.6f}    [ {st_res['wald']['low']:.6f}, {st_res['wald']['high']:.6f} ]")

        # 3) Confronto SAME-STREAM (chiave per “mele con mele”)
        ss_res = compare_same_stream(
            model, device, test_loader, clean_by_batch,
            baseline_hist, baseline_dist, num_classes,
            all_faults, N, eps=EPS, conf=CONF, block=BLOCK, seed=SEED+42, N_pop=None
        )

        # Scrivi un file riassunto per ogni N
        out_path = os.path.join(base_dir, f"{dataset}_{net}_N{N}_compare.txt")
        with open(out_path, "w") as f:
            f.write(f"=== N={N} Comparison (EP vs STAT vs SAME-STREAM) ===\n\n")

            f.write("-- EP stop --\n")
            f.write(f"n_ep={ep_res['n']}  p_hat={ep_res['p_hat']:.8f}\n")
            f.write(f"Wald   half={ep_res['wald']['half']:.8f}  [{ep_res['wald']['low']:.8f}, {ep_res['wald']['high']:.8f}]\n")
            f.write(f"Wilson half={ep_res['wilson']['half']:.8f}  [{ep_res['wilson']['low']:.8f}, {ep_res['wilson']['high']:.8f}]\n\n")

            f.write("-- STAT stop --\n")
            f.write(f"n_st={st_res['n']}  p_hat={st_res['p_hat']:.8f}\n")
            f.write(f"Wilson half={st_res['wilson']['half']:.8f}  [{st_res['wilson']['low']:.8f}, {st_res['wilson']['high']:.8f}]\n")
            f.write(f"Wald   half={st_res['wald']['half']:.8f}    [{st_res['wald']['low']:.8f}, {st_res['wald']['high']:.8f}]\n\n")

            if "stop_wald" in ss_res:
                sw = ss_res["stop_wald"]
                f.write("-- SAME-STREAM @ WALD stop --\n")
                f.write(f"n={sw['n']}  p_hat={sw['p_hat']:.8f}\n")
                f.write(f"Wald   half={sw['wald']['half']:.8f}  [{sw['wald']['low']:.8f}, {sw['wald']['high']:.8f}]\n")
                f.write(f"Wilson half={sw['wilson']['half']:.8f}  [{sw['wilson']['low']:.8f}, {sw['wilson']['high']:.8f}]\n\n")

            if "stop_wilson" in ss_res:
                sw = ss_res["stop_wilson"]
                f.write("-- SAME-STREAM @ WILSON stop --\n")
                f.write(f"n={sw['n']}  p_hat={sw['p_hat']:.8f}\n")
                f.write(f"Wilson half={sw['wilson']['half']:.8f}  [{sw['wilson']['low']:.8f}, {sw['wilson']['high']:.8f}]\n")
                f.write(f"Wald   half={sw['wald']['half']:.8f}    [{sw['wald']['low']:.8f}, {sw['wald']['high']:.8f}]\n")

        print(f"\n[WRITE] Saved comparison to: {out_path}")
