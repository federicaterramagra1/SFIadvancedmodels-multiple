#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import random
import heapq
from itertools import islice, product, combinations
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from faultManager.WeightFault import WeightFault
from faultManager.WeightFaultInjector import WeightFaultInjector
from utils import get_loader, load_from_dict, get_network
import SETTINGS


# ============================= Utils numerici & stampa sicura =============================

def _sci_format_comb(n: int, k: int) -> str:
    """
    Formatta C(n,k) in notazione scientifica *senza* materializzare l'intero.
    Usa log10 via funzioni gamma (lgamma) per evitare overflow.
    """
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
    """
    Ritorna il tensore dei pesi quantizzati in modo compatibile
    con varie versioni di PyTorch (API pubblica e fallback).
    """
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
    as_list=True => materializza in lista (ok per MLP piccoli/medi).
    as_list=False => ritorna un generatore (per reti grandi).
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


# ============================= Sampler & statistica (PROPOSED) =============================

def srs_combinations(pool, r, seed=None, max_yield=None):
    """
    SRS i.i.d. sulle combinazioni:
    - ad ogni iterazione estrae r indici distinti (senza ripetizione dentro la combo)
    - tra iterazioni si accettano ripetizioni (campionamento con reinserimento sull'insieme delle combinazioni)
    - questo rende gli FR i.i.d.
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


def _z_from_conf(conf: float) -> float:
    # Per 99.9% puoi usare 3.291 se necessario
    table = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576, 0.999: 3.291}
    return table.get(conf, 1.96)


def fpc_factor(N_pop: Optional[int], n: int) -> float:
    """
    FPC = sqrt((N_pop - n)/(N_pop - 1)).
    Usa N_pop=None per 'infinito' o per campionamento con reinserimento -> FPC=1.
    """
    if not N_pop or N_pop <= 0:
        return 1.0
    if n <= 1 or n >= N_pop:
        val = max(0.0, (N_pop - n) / max(1, N_pop - 1))
        return math.sqrt(val)
    return math.sqrt((N_pop - n) / (N_pop - 1))


def halfwidth_normal_fpc(p_hat: float, n: int, conf: float, N_pop: Optional[int]) -> float:
    """
    Half-width (normale) per proporzione con FPC opzionale.
    """
    z = _z_from_conf(conf)
    p = min(max(p_hat, 1e-12), 1.0 - 1e-12)
    hw = z * math.sqrt(p * (1.0 - p) / max(1, n))
    return hw * fpc_factor(N_pop, n)


def plan_n_with_fpc(p_hat: float, e_target: float, conf: float, N_pop: Optional[int]) -> int:
    """
    Pianifica la numerosità TOTALE desiderata dato e_target e p_hat (FPC opzionale).
    Se N_pop=None -> formula senza FPC (pop. 'infinita' o sampling con reinserimento).
    """
    z = _z_from_conf(conf)
    p = min(max(p_hat, 1e-6), 1.0 - 1e-6)
    if not N_pop or N_pop <= 0:
        # caso infinito (o con reinserimento): n >= z^2 p(1-p)/e^2
        n = math.ceil((z * z * p * (1.0 - p)) / (e_target * e_target))
        return max(1, n)
    # Eq. (4) con FPC
    denom = 1.0 + (e_target * e_target * (N_pop - 1.0)) / (z * z * p * (1.0 - p))
    n = N_pop / denom
    return int(min(N_pop, max(1, math.ceil(n))))


def ep_next_error(e_goal: float, E_hat: float, p_hat: float) -> float:
    """
    E–P curve del paper: errore target al prossimo giro in funzione di p_hat e dell'errore attuale E_hat.
    Se E_hat/3 <= e_goal -> torna e_goal.
    Altrimenti parabola per (0,e_goal), (1,e_goal), (0.5, E_hat/3).
    """
    if E_hat <= 0.0:
        return e_goal
    thresh = E_hat / 3.0
    if thresh <= e_goal:
        return e_goal
    k = 4.0 * (thresh - e_goal)
    val = -k * (p_hat ** 2) + k * p_hat + e_goal
    return float(min(max(val, e_goal), thresh))


def wald_ci(p_hat: float, n: int, conf: float, N_pop: Optional[int]):
    """
    Intervallo di Wald con eventuale FPC.
    Restituisce (low, high, half, se_effettiva).
    """
    half = halfwidth_normal_fpc(p_hat, n, conf, N_pop)
    z = _z_from_conf(conf)
    se = half / max(z, 1e-12)
    low = max(0.0, p_hat - half)
    high = min(1.0, p_hat + half)
    return low, high, half, se


# ============================= Build, quantize, clean pass =============================

def build_and_quantize_once():
    # Enforce CPU quantization path (x86)
    torch.backends.quantized.engine = "fbgemm"
    device = torch.device("cpu")

    # 1) Build model + load ckpt (se presente)
    model = get_network(SETTINGS.NETWORK_NAME, device, SETTINGS.DATASET_NAME)
    model.to(device).eval()
    ckpt = f"./trained_models/{SETTINGS.DATASET_NAME}_{SETTINGS.NETWORK_NAME}_trained.pth"
    if os.path.exists(ckpt):
        load_from_dict(model, device, ckpt)
        print("modello caricato")
    else:
        print(f"checkpoint non trovato: {ckpt} (proseguo senza)")

    # 2) Loader (train per calibrazione quantize, test per inferenza)
    train_loader, _, test_loader = get_loader(
        dataset_name=SETTINGS.DATASET_NAME,
        batch_size=SETTINGS.BATCH_SIZE,
        network_name=SETTINGS.NETWORK_NAME
    )

    # 3) Quantize (se supportato)
    if hasattr(model, "quantize_model"):
        model.to(device)
        model.quantize_model(calib_loader=train_loader)
        model.eval()
        print("quantizzazione completata")
    else:
        print("modello non quantizzabile (salto quantizzazione)")

    # 4) Clean predictions AFTER quantization: lista per-batch
    clean_by_batch = []
    with torch.inference_mode():
        for xb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            clean_by_batch.append(torch.argmax(logits, dim=1).cpu())

    # 4b) Baseline distribuzione predizioni (per bias)
    clean_flat = torch.cat(clean_by_batch, dim=0)
    num_classes = int(clean_flat.max().item()) + 1 if clean_flat.numel() > 0 else 2
    baseline_hist = np.bincount(clean_flat.numpy(), minlength=num_classes)
    baseline_dist = baseline_hist / max(1, baseline_hist.sum())
    print(f"[BASELINE] pred dist = {baseline_dist}")

    # 5) Micro-check: 3 injection singole su un batch, con restore garantito
    injector = WeightFaultInjector(model)
    preview = list(islice(_build_all_faults(model, as_list=False), 3))
    if len(preview) == 0:
        raise RuntimeError("Nessun fault enumerato: verifica che il modello sia realmente quantizzato.")

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

    # 6) Conteggio siti di fault
    all_faults = _build_all_faults(model, as_list=True)
    print(f"Siti single-bit totali: {len(all_faults)}")
    if len(all_faults) == 0:
        raise RuntimeError("Nessun fault enumerato: verifica quantizzazione e layer target.")

    return model, device, test_loader, clean_by_batch, baseline_hist, baseline_dist, num_classes


# ============================= Valutazione singola combo =============================

def _evaluate_combo(model, device, test_loader, clean_by_batch, injector, combo, inj_id,
                    total_samples, baseline_hist, baseline_dist, num_classes):
    """
    Inietta una combo, valuta FRcrit + metriche di bias, ripristina.
    Ritorna (frcrit, faults, bias, fault_hist, mism_by_clean, cnt_by_clean, cm_cf)
    """
    faults = [WeightFault(injection=inj_id, layer_name=ln, tensor_index=ti, bits=[bt])
              for (ln, ti, bt) in combo]
    try:
        injector.inject_faults(faults, 'bit-flip')

        mismatches = 0
        fault_hist = np.zeros_like(baseline_hist, dtype=np.int64)
        mism_by_clean = np.zeros(num_classes, dtype=np.int64)
        cnt_by_clean  = np.zeros(num_classes, dtype=np.int64)
        cm_cf = np.zeros((num_classes, num_classes), dtype=np.int64)  # clean -> fault

        with torch.inference_mode():
            for (batch_i, (xb, _)) in enumerate(test_loader):
                xb = xb.to(device)
                logits_f = model(xb)
                pred_f = torch.argmax(logits_f, dim=1).cpu().numpy()
                np.add.at(fault_hist, pred_f, 1)

                clean_pred = clean_by_batch[batch_i].numpy()
                mism = (pred_f != clean_pred)
                mismatches += int(mism.sum())

                # Confusione clean -> fault (multi-classe)
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

        # Simmetria dei flip (multi-classe): 0 = simmetrici, 1 = altamente asimmetrici
        off_sum = int(cm_cf.sum() - np.trace(cm_cf))
        asym_num = 0
        if num_classes >= 2:
            diff = np.abs(cm_cf - cm_cf.T)
            asym_num = int(diff[np.triu_indices(num_classes, k=1)].sum())
        flip_asym = float(asym_num) / max(1, off_sum)  # [0,1]
        agree = float(np.trace(cm_cf)) / max(1, int(cm_cf.sum()))

        bias = {
            "maj_cls": maj_cls,
            "maj_share": maj_share,
            "delta_max": delta_max,
            "kl": kl,
            "flip_asym": flip_asym,
            "agree": agree
        }

    finally:
        injector.restore_golden()

    return frcrit, faults, bias, fault_hist, mism_by_clean, cnt_by_clean, cm_cf


# ============================= Campagna PROPOSED (iterativa EP+FPC) =============================

def run_statistical_iterative_ep_fpc_campaign(
    model, device, test_loader, clean_by_batch, all_faults, N,
    baseline_hist, baseline_dist, num_classes,
    pilot=None, e_goal=None, conf=None, block=None, budget_cap=None, seed=None,
    use_fpc: bool = False, N_pop: Optional[int] = None,
    save_dir="results_minimal", dataset_name=None, net_name=None, bs=None, prefix_tag="_EPFPC"
):
    """
    Campagna SFI iterativa 'proposed':
      - pilot -> stima p_hat e E_hat (half-width normale con FPC opzionale)
      - E–P curve -> definisce e_next
      - plan_n_with_fpc(p_hat, e_next) -> definisce n_totale desiderato
      - inietta fino a n_totale; aggiorna p_hat, E_hat; stop quando E_hat <= e_goal o budget
    Note:
      - Se campioni con reinserimento (come srs_combinations), lascia use_fpc=False e N_pop=None.
      - Se passi a SRS senza reinserimento su una lista finita, usa use_fpc=True e fornisci N_pop.
    """
    # ----- Parametri -----
    pilot   = pilot   if pilot   is not None else getattr(SETTINGS, "STAT_PILOT", 200)
    e_goal  = e_goal  if e_goal  is not None else getattr(SETTINGS, "STAT_EPS",   0.005)
    conf    = conf    if conf    is not None else getattr(SETTINGS, "STAT_CONF",  0.95)
    block   = block   if block   is not None else getattr(SETTINGS, "STAT_BLOCK", 50)
    budget_cap = budget_cap if budget_cap is not None else getattr(SETTINGS, "STAT_BUDGET_CAP", None)
    seed    = seed    if seed    is not None else getattr(SETTINGS, "SEED", 0)

    if not use_fpc:
        N_pop = None  # FPC disattivata

    injector = WeightFaultInjector(model)
    total_samples = sum(len(t) for t in clean_by_batch)
    if total_samples == 0:
        raise RuntimeError("Test loader senza campioni.")

    # ----- Aggregatori globali -----
    global_fault_hist = np.zeros_like(baseline_hist, dtype=np.int64)
    mism_by_clean_sum = np.zeros(num_classes, dtype=np.int64)
    cnt_by_clean_sum  = np.zeros(num_classes, dtype=np.int64)
    global_cm_cf      = np.zeros((num_classes, num_classes), dtype=np.int64)
    maj_shares, kls   = [], []

    # Tracking media e varianza delle FRcrit per-injection (Welford)
    mean_frcrit = 0.0
    m2_frcrit = 0.0
    count_fr = 0

    # ----- Path output -----
    dataset = dataset_name or SETTINGS.DATASET_NAME
    net     = net_name    or SETTINGS.NETWORK_NAME
    save_path = os.path.join(save_dir, dataset, net, f"batch_{bs}", "minimal_iterEP")
    os.makedirs(save_path, exist_ok=True)
    prefix = f"{dataset}_{net}_PROP_N{N}_batch{bs}{prefix_tag}"
    output_file = os.path.join(save_path, f"{prefix}.txt")

    # ----- Generatori SRS -----
    gen = srs_combinations(all_faults, r=N, seed=seed,   max_yield=pilot)
    gen_more = srs_combinations(all_faults, r=N, seed=seed+1, max_yield=None)

    # ----- Pilot -----
    sum_fr, n_inj, inj_id = 0.0, 0, 0
    top_heap = []  # (frcrit, inj_id, faults, bias)

    pbar = tqdm(gen, total=pilot, desc=f"[PROP] pilot N={N}")
    for combo in pbar:
        inj_id += 1
        frcrit, faults, bias, fh, mbc, cbc, cm = _evaluate_combo(
            model, device, test_loader, clean_by_batch, injector, combo, inj_id, total_samples,
            baseline_hist, baseline_dist, num_classes
        )
        sum_fr += frcrit
        n_inj  += 1

        # Welford update
        count_fr += 1
        delta = frcrit - mean_frcrit
        mean_frcrit += delta / count_fr
        m2_frcrit += delta * (frcrit - mean_frcrit)

        # aggregazione globale
        global_fault_hist += fh
        mism_by_clean_sum += mbc
        cnt_by_clean_sum  += cbc
        global_cm_cf      += cm
        maj_shares.append(bias["maj_share"])
        kls.append(bias["kl"])

        # top-100
        if len(top_heap) < 100:
            heapq.heappush(top_heap, (frcrit, inj_id, faults, bias))
        elif frcrit > top_heap[0][0]:
            heapq.heapreplace(top_heap, (frcrit, inj_id, faults, bias))

    # Stime cumulative dopo pilot
    p_hat = (sum_fr / n_inj) if n_inj else 0.0
    E_hat = halfwidth_normal_fpc(p_hat, n_inj, conf, N_pop)

    # ----- Loop iterativo -----
    steps = 0
    while True:
        steps += 1
        # Stop se già entro il target
        if E_hat <= e_goal:
            break
        if budget_cap and n_inj >= budget_cap:
            break

        # E–P curve -> prossimo errore target
        e_next = ep_next_error(e_goal=e_goal, E_hat=E_hat, p_hat=p_hat)

        # Pianifica numerosità totale desiderata a fine prossimo step
        n_tot_desired = plan_n_with_fpc(p_hat=p_hat, e_target=e_next, conf=conf, N_pop=N_pop)
        add_needed = max(0, n_tot_desired - n_inj)
        if budget_cap:
            add_needed = min(add_needed, max(0, budget_cap - n_inj))

        # Iniezioni aggiuntive in blocchi per progress report
        to_do = add_needed
        pbar2 = tqdm(total=add_needed, desc=f"[PROP] step#{steps} to n={n_tot_desired} (e_next={e_next:.6g})")
        while to_do > 0:
            combo = next(gen_more)
            inj_id += 1
            frcrit, faults, bias, fh, mbc, cbc, cm = _evaluate_combo(
                model, device, test_loader, clean_by_batch, injector, combo, inj_id, total_samples,
                baseline_hist, baseline_dist, num_classes
            )
            sum_fr += frcrit
            n_inj  += 1
            to_do  -= 1
            pbar2.update(1)

            # Welford update
            count_fr += 1
            delta = frcrit - mean_frcrit
            mean_frcrit += delta / count_fr
            m2_frcrit += delta * (frcrit - mean_frcrit)

            # aggregatori
            global_fault_hist += fh
            mism_by_clean_sum += mbc
            cnt_by_clean_sum  += cbc
            global_cm_cf      += cm
            maj_shares.append(bias["maj_share"])
            kls.append(bias["kl"])

            # top-100
            if len(top_heap) < 100:
                heapq.heappush(top_heap, (frcrit, inj_id, faults, bias))
            elif frcrit > top_heap[0][0]:
                heapq.heapreplace(top_heap, (frcrit, inj_id, faults, bias))

            # checkpoint ogni 'block' per valutare stop anticipato
            if (n_inj % block) == 0:
                p_hat = (sum_fr / n_inj)
                E_hat = halfwidth_normal_fpc(p_hat, n_inj, conf, N_pop)
                if E_hat <= e_goal or (budget_cap and n_inj >= budget_cap):
                    break

        # aggiorna stime a fine step
        p_hat = (sum_fr / n_inj)
        E_hat = halfwidth_normal_fpc(p_hat, n_inj, conf, N_pop)

        if E_hat <= e_goal or (budget_cap and n_inj >= budget_cap):
            break

    # ----- Report finale -----
    avg_frcrit = (sum_fr / n_inj) if n_inj else 0.0
    half_norm = halfwidth_normal_fpc(avg_frcrit, n_inj, conf, N_pop)
    w_low, w_high, w_half, w_se = wald_ci(avg_frcrit, n_inj, conf, N_pop)
    sample_std = math.sqrt(m2_frcrit / (count_fr - 1)) if count_fr > 1 else 0.0

    # global summary (come nelle tue funzioni)
    total_preds = int(global_fault_hist.sum())
    global_fault_dist = (global_fault_hist / total_preds) if total_preds > 0 else baseline_dist
    tiny = 1e-12
    global_delta_max = float(np.max(np.abs(global_fault_dist - baseline_dist)))
    global_kl = float(np.sum(global_fault_dist * np.log((global_fault_dist + tiny)/(baseline_dist + tiny))))
    global_tv = 0.5 * float(np.sum(np.abs(global_fault_dist - baseline_dist)))
    H = lambda p: float(-np.sum(p * np.log(p + tiny)))
    entropy_baseline = H(baseline_dist)
    entropy_global   = H(global_fault_dist)
    entropy_drop     = entropy_baseline - entropy_global
    ber_per_class = []
    for c in range(num_classes):
        ber_c = (mism_by_clean_sum[c] / max(1, cnt_by_clean_sum[c]))
        ber_per_class.append(float(ber_c))
    BER = float(np.mean(ber_per_class)) if ber_per_class else 0.0
    agree_global = float(np.trace(global_cm_cf)) / max(1, int(global_cm_cf.sum()))
    off_sum = int(global_cm_cf.sum() - np.trace(global_cm_cf))
    diff = np.abs(global_cm_cf - global_cm_cf.T)
    asym_num = int(diff[np.triu_indices(num_classes, k=1)].sum())
    flip_asym_global = float(asym_num) / max(1, off_sum)
    maj_shares_arr = np.array(maj_shares) if maj_shares else np.array([])
    mean_share = float(maj_shares_arr.mean()) if maj_shares_arr.size else 0.0
    p90_share  = float(np.percentile(maj_shares_arr, 90)) if maj_shares_arr.size else 0.0
    frac_collapse_080 = float(np.mean(maj_shares_arr >= 0.80)) if maj_shares_arr.size else 0.0
    mean_kl = float(np.mean(kls)) if kls else 0.0

    top_sorted = sorted(top_heap, key=lambda x: x[0], reverse=True)
    with open(output_file, "w") as f:
        f.write(f"[PROP] N={N}  FRcrit_avg={avg_frcrit:.8f}  conf={conf}  steps={steps}  n={n_inj}\n")
        f.write(f"half_norm={half_norm:.8f}  e_goal={e_goal:.8f}  use_fpc={use_fpc}  N_pop={N_pop}\n")
        # --- Nuove righe: Intervallo di Wald e deviazione standard ---
        f.write(f"Wald_CI({int(conf*100)}%): [{w_low:.8f}, {w_high:.8f}]  half={w_half:.8f}  se={w_se:.8f}  FPC={'on' if (use_fpc and (N_pop or 0)>0) else 'off'}\n")
        f.write(f"SampleStdDev_FRcrit_across_injections: s={sample_std:.8f} (n={n_inj})\n")
        # --------------------------------------------------------------
        f.write(f"injections_used={n_inj}  pilot={pilot}  block={block}  budget_cap={budget_cap}\n")
        f.write(f"baseline_pred_dist={baseline_dist.tolist()}\n")
        f.write(
            "global_summary_over_injections: "
            f"fault_pred_dist={global_fault_dist.tolist()} "
            f"Δmax={global_delta_max:.3f} KL={global_kl:.3f} TV={global_tv:.3f} "
            f"H_baseline={entropy_baseline:.3f} H_global={entropy_global:.3f} ΔH={entropy_drop:.3f} "
            f"BER={BER:.4f} per_class={ber_per_class} "
            f"agree={agree_global:.3f} flip_asym={flip_asym_global:.3f}\n"
        )
        f.write(
            f"bias_frequency: mean_share={mean_share:.3f} p90_share={p90_share:.3f} "
            f"mean_KL={mean_kl:.3f} frac_collapse(share≥0.80)={frac_collapse_080:.3f}\n\n"
        )
        f.write(f"Top-{min(100, len(top_sorted))} worst injections (proposed iter EP+FPC)\n")
        for rank, (frcrit, inj, faults, bias) in enumerate(top_sorted, 1):
            desc = ", ".join(f"{ft.layer_name}[{ft.tensor_index}] bit{ft.bits[0]}" for ft in faults)
            f.write(
                f"{rank:3d}) Inj {inj:6d} | FRcrit={frcrit:.6f} | "
                f"maj={bias['maj_cls']}@{bias['maj_share']:.2f} Δmax={bias['delta_max']:.3f} "
                f"KL={bias['kl']:.3f} | {desc}\n"
            )
                

    print(f"[PROP] N={N}  avgFRcrit={avg_frcrit:.6f}  half_norm={half_norm:.6f}  n={n_inj} → {output_file}")
    return avg_frcrit, half_norm, n_inj, steps, top_sorted, output_file


# ============================= Dispatcher: ESAUSTIVA oppure PROPOSED =============================

def run_fault_injection(
    model,
    device,
    test_loader,
    clean_by_batch,            # lista di tensori predetti "clean" per ogni batch (in ordine)
    baseline_hist,
    baseline_dist,
    num_classes,
    N,
    MAX_FAULTS=4_000_000,
    save_dir="results_minimal",
    seed=None,
    exhaustive_up_to_n=3,       # forzare ESAUSTIVA per N <= 3
    use_fpc: bool = False,      # FPC off di default (campionamento con reinserimento)
    N_pop: Optional[int] = None # se use_fpc=True, passa la size della popolazione finita
):
    """
    Campagna:
      - ESAUSTIVA se (N <= exhaustive_up_to_n) oppure (C(num_faults, N) <= MAX_FAULTS),
      - altrimenti SFI iterativa PROPOSED (EP+FPC opzionale).
    Salva Top-100 + metriche salienti in .txt.
    """
    t0 = time.time()

    # Output path base
    dataset = SETTINGS.DATASET_NAME
    net_name = SETTINGS.NETWORK_NAME
    bs = getattr(test_loader, "batch_size", None) or SETTINGS.BATCH_SIZE
    save_path = os.path.join(save_dir, dataset, net_name, f"batch_{bs}", "minimal")
    os.makedirs(save_path, exist_ok=True)

    # Fault elementari
    all_faults = _build_all_faults(model, as_list=True)
    num_faults = len(all_faults)
    if num_faults == 0:
        raise RuntimeError("Lista fault vuota: impossibile proseguire.")

    if N > num_faults:
        print(f"[WARN] N={N} > num_faults={num_faults}. Ridimensiono N a {num_faults}.")
        N = num_faults

    # ESAUSTIVA per N <= exhaustive_up_to_n (sempre)
    force_exhaustive = (N <= exhaustive_up_to_n)
    total_possible = math.comb(num_faults, N)

    if force_exhaustive or total_possible <= MAX_FAULTS:
        # ===== ESAUSTIVA =====
        print(f"[INFO] ESAUSTIVA (N={N}, combinazioni={_sci_format_comb(num_faults, N)})...")
        fault_combos = combinations(all_faults, N)
        max_iters = total_possible

        injector = WeightFaultInjector(model)
        TOP_K = 100
        top_heap = []  # min-heap di tuple (frcrit, inj_id, faults, bias)
        sum_fr, n_injected = 0.0, 0
        total_samples = sum(len(t) for t in clean_by_batch)
        if total_samples == 0:
            raise RuntimeError("Test loader senza campioni.")

        # Aggregatori globali anche per esaustiva
        global_fault_hist = np.zeros_like(baseline_hist, dtype=np.int64)
        mism_by_clean_sum = np.zeros(num_classes, dtype=np.int64)
        cnt_by_clean_sum  = np.zeros(num_classes, dtype=np.int64)
        global_cm_cf = np.zeros((num_classes, num_classes), dtype=np.int64)
        maj_shares, kls = [], []

        # Tracking std dev FRcrit per-injection
        mean_frcrit = 0.0
        m2_frcrit = 0.0
        count_fr = 0

        pbar = tqdm(fault_combos, total=max_iters, desc=f"N={N} (exa)")
        for inj_id, combo in enumerate(pbar, 1):
            frcrit, faults, bias, fh, mbc, cbc, cm = _evaluate_combo(
                model, device, test_loader, clean_by_batch, injector, combo, inj_id, total_samples,
                baseline_hist, baseline_dist, num_classes
            )
            sum_fr += frcrit
            n_injected += 1

            # Welford update
            count_fr += 1
            delta = frcrit - mean_frcrit
            mean_frcrit += delta / count_fr
            m2_frcrit += delta * (frcrit - mean_frcrit)

            # aggregazione globale
            global_fault_hist += fh
            mism_by_clean_sum += mbc
            cnt_by_clean_sum  += cbc
            global_cm_cf += cm
            maj_shares.append(bias["maj_share"])
            kls.append(bias["kl"])

            if len(top_heap) < TOP_K:
                heapq.heappush(top_heap, (frcrit, inj_id, faults, bias))
            elif frcrit > top_heap[0][0]:
                heapq.heapreplace(top_heap, (frcrit, inj_id, faults, bias))

        avg_frcrit = (sum_fr / n_injected) if n_injected else 0.0
        prefix = f"{dataset}_{net_name}_minimal_N{N}_batch{bs}"
        output_file = os.path.join(save_path, f"{prefix}.txt")
        top_sorted = sorted(top_heap, key=lambda x: x[0], reverse=True)

        # Riepilogo globale esaustiva
        total_preds = int(global_fault_hist.sum())
        global_fault_dist = (global_fault_hist / total_preds) if total_preds > 0 else baseline_dist
        eps = 1e-12
        global_delta_max = float(np.max(np.abs(global_fault_dist - baseline_dist)))
        global_kl = float(np.sum(global_fault_dist * np.log((global_fault_dist + eps)/(baseline_dist + eps))))
        global_tv = 0.5 * float(np.sum(np.abs(global_fault_dist - baseline_dist)))
        H = lambda p: float(-np.sum(p * np.log(p + eps)))
        entropy_baseline = H(baseline_dist)
        entropy_global   = H(global_fault_dist)
        entropy_drop = entropy_baseline - entropy_global
        ber_per_class = []
        for c in range(num_classes):
            ber_c = (mism_by_clean_sum[c] / max(1, cnt_by_clean_sum[c]))
            ber_per_class.append(float(ber_c))
        BER = float(np.mean(ber_per_class)) if ber_per_class else 0.0
        agree_global = float(np.trace(global_cm_cf)) / max(1, int(global_cm_cf.sum()))
        off_sum = int(global_cm_cf.sum() - np.trace(global_cm_cf))
        diff = np.abs(global_cm_cf - global_cm_cf.T)
        asym_num = int(diff[np.triu_indices(num_classes, k=1)].sum())
        flip_asym_global = float(asym_num) / max(1, off_sum)
        maj_shares_arr = np.array(maj_shares) if maj_shares else np.array([])
        mean_share = float(maj_shares_arr.mean()) if maj_shares_arr.size else 0.0
        p90_share  = float(np.percentile(maj_shares_arr, 90)) if maj_shares_arr.size else 0.0
        frac_collapse_080 = float(np.mean(maj_shares_arr >= 0.80)) if maj_shares_arr.size else 0.0
        mean_kl = float(np.mean(kls)) if kls else 0.0

        # --- Nuovo: intervallo di Wald e std dev per ESAUSTIVA ---
        conf = getattr(SETTINGS, "STAT_CONF", 0.95)
        # Se esaustiva, la "popolazione finita" è total_possible -> FPC azzera halfwidth
        w_low, w_high, w_half, w_se = wald_ci(avg_frcrit, n_injected, conf, N_pop=total_possible)
        sample_std = math.sqrt(m2_frcrit / (count_fr - 1)) if count_fr > 1 else 0.0
        # ----------------------------------------------------------

        with open(output_file, "w") as f:
            f.write(f"Top-{min(TOP_K, len(top_sorted))} worst injections  (N={N})\n")
            f.write(f"Failure Rate (critical): {avg_frcrit:.8f}  on {n_injected} injections\n")
            # Aggiunta delle statistiche richieste
            f.write(f"Wald_CI({int(conf*100)}%): [{w_low:.8f}, {w_high:.8f}]  half={w_half:.8f}  se={w_se:.8f}  FPC={'on (exhaustive)'}\n")
            f.write(f"SampleStdDev_FRcrit_across_injections: s={sample_std:.8f} (n={n_injected})\n")
            # Fine aggiunta
            f.write(f"baseline_pred_dist={baseline_dist.tolist()}\n")
            f.write(
                "global_summary_over_injections: "
                f"fault_pred_dist={global_fault_dist.tolist()} "
                f"Δmax={global_delta_max:.3f} KL={global_kl:.3f} TV={global_tv:.3f} "
                f"H_baseline={entropy_baseline:.3f} H_global={entropy_global:.3f} ΔH={entropy_drop:.3f} "
                f"BER={BER:.4f} per_class={ber_per_class} "
                f"agree={agree_global:.3f} flip_asym={flip_asym_global:.3f}\n"
            )
            f.write(
                f"bias_frequency: mean_share={mean_share:.3f} p90_share={p90_share:.3f} "
                f"mean_KL={mean_kl:.3f} frac_collapse(share≥0.80)={frac_collapse_080:.3f}\n\n"
            )
            for rank, (frcrit, inj_id, faults, bias) in enumerate(top_sorted, 1):
                desc = ", ".join(f"{ft.layer_name}[{ft.tensor_index}] bit{ft.bits[0]}" for ft in faults)
                f.write(
                    f"{rank:3d}) Inj {inj_id:6d} | FRcrit={frcrit:.6f} | "
                    f"maj={bias['maj_cls']}@{bias['maj_share']:.2f} Δmax={bias['delta_max']:.3f} "
                    f"KL={bias['kl']:.3f} | {desc}\n"
                )

        dt_min = (time.time() - t0) / 60.0
        print(f"[EXA] salvato {output_file} – {dt_min:.2f} min (avg FRcrit={avg_frcrit:.6f}, injections={n_injected})")
        return avg_frcrit, None, n_injected, top_sorted, output_file

    else:
        # ===== STATISTICA PROPOSED (EP+FPC opzionale) =====
        comb_str = _sci_format_comb(num_faults, N)
        print(f"[INFO] STATISTICA (PROPOSED): C({num_faults},{N})≈{comb_str} > MAX_FAULTS={MAX_FAULTS}. "
              f"Avvio iterativo EP{' + FPC' if use_fpc else ''}.")
        avg_frcrit, half_norm, n_used, steps, top_sorted, out_file = run_statistical_iterative_ep_fpc_campaign(
            model, device, test_loader, clean_by_batch, all_faults, N,
            baseline_hist, baseline_dist, num_classes,
            pilot=getattr(SETTINGS, "STAT_PILOT", 200),
            e_goal=getattr(SETTINGS, "STAT_EPS",   0.005),
            conf=getattr(SETTINGS, "STAT_CONF",    0.95),
            block=getattr(SETTINGS, "STAT_BLOCK",  50),
            budget_cap=getattr(SETTINGS, "STAT_BUDGET_CAP", None),
            seed=seed if seed is not None else getattr(SETTINGS, "SEED", 0),
            use_fpc=use_fpc, N_pop=N_pop,
            save_dir=save_dir, dataset_name=dataset, net_name=net_name, bs=bs
        )
        dt_min = (time.time() - t0) / 60.0
        print(f"[PROP] salvato {out_file} – {dt_min:.2f} min (avg FRcrit={avg_frcrit:.6f}, half_norm={half_norm:.6f}, n={n_used}, steps={steps})")
        return avg_frcrit, (half_norm,), n_used, top_sorted, out_file


# =================================== Main ===================================

# =================================== Main ===================================

if __name__ == "__main__":
    # Build + quantize + clean pass
    model, device, test_loader, clean_by_batch, baseline_hist, baseline_dist, num_classes = build_and_quantize_once()

    # STATISTICHE EP per N = 1..5 (NO esaustiva)
    for N in [#1, 2,3,4,5, 6, 7, 8, 9, 10, 50, 100, 150, 
             #384,
               576, 768, 960, 1104, 1408, 1728, 1984, 2048, 2208 #, 384,576,
              #766 ,960, 1104, 1408,1728,1984,2048,2208
]:
        avg, half_tuple, n_used, _, out_file = run_fault_injection(
            model=model,
            device=device,
            test_loader=test_loader,
            clean_by_batch=clean_by_batch,
            baseline_hist=baseline_hist,
            baseline_dist=baseline_dist,
            num_classes=num_classes,
            N=N,
            MAX_FAULTS=0,             # forza il ramo statistico (mai esaustiva)
            save_dir="results_minimal",
            seed=getattr(SETTINGS, "SEED", 0),
            exhaustive_up_to_n=-1,    # disattiva ESA anche per N piccoli
            use_fpc=False,            # campionamento con reinserimento => niente FPC
            N_pop=None
        )
        # half_tuple può essere None nel ramo esaustivo, qui forziamo sempre statistico (come sopra)
        half = half_tuple[0] if isinstance(half_tuple, (list, tuple)) else None
        print(f"[EP] N={N} | FR={avg:.6g} | half≈{(half if half is not None else float('nan')):.6g} | injections={n_used} | file={out_file}")
