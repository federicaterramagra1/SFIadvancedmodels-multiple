#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import random
import heapq
from itertools import islice, product, combinations

import numpy as np
import torch
from tqdm import tqdm

from faultManager.WeightFault import WeightFault
from faultManager.WeightFaultInjector import WeightFaultInjector
from utils import get_loader, load_from_dict, get_network, load_quantized_model, save_quantized_model

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
    # log10(C(n,k)) = (lgamma(n+1)-lgamma(k+1)-lgamma(n-k+1))/ln(10)
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


# ============================= Sampler & statistica =============================

def srs_combinations(pool, r, seed=None, max_yield=None):
    """
    SRS i.i.d. sulle combinazioni:
    - ad ogni iterazione estrae r indici distinti (senza ripetizione dentro la combo)
    - tra iterazioni si accettano ripetizioni (campionamento con reinserimento sull'insieme delle combinazioni)
    - questo rende gli FR i.i.d. => CI standard robuste.
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


def wilson_ci(p_hat, n, conf=0.95):
    """
    Intervallo di confidenza di Wilson per proporzioni.
    Ritorna (low, high, halfwidth).
    """
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(conf, 1.96)
    if n == 0:
        return 0.0, 1.0, 0.5
    p = min(max(p_hat, 1e-12), 1 - 1e-12)
    denom = 1 + (z*z)/n
    center = (p + (z*z)/(2*n)) / denom
    half = (z * math.sqrt((p*(1-p)/n) + (z*z)/(4*n*n))) / denom
    low = max(0.0, center - half)
    high = min(1.0, center + half)
    return low, high, half


def choose_n_for_ci_normal(p_hat, eps=0.005, conf=0.95):
    """
    Stima iniziale della numerosità con approssimazione normale.
    Usata per fissare un target 'ragionevole' prima dello stop sequenziale con Wilson.
    """
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(conf, 1.96)
    p = min(max(p_hat, 1e-4), 1 - 1e-4)  # evita n enormi se p_hat=0
    return math.ceil((z*z*p*(1-p))/(eps*eps))


# ============================= Build, quantize, clean pass =============================

def build_and_quantize_once():
    # CPU quant path (x86)
    torch.backends.quantized.engine = "fbgemm"
    device = torch.device("cpu")

    ds_name = getattr(SETTINGS, "DATASET_NAME", getattr(SETTINGS, "DATASET", ""))
    net_name = getattr(SETTINGS, "NETWORK_NAME", getattr(SETTINGS, "NETWORK", ""))

    # Loader (test sempre; train solo se devo calibrare)
    train_loader, _, test_loader = get_loader(
        dataset_name=ds_name,
        batch_size=getattr(SETTINGS, "BATCH_SIZE", 64),
        network_name=net_name
    )

    # 1) Prova a caricare il quantizzato
    qmodel, qpath = load_quantized_model(ds_name, net_name, device="cpu", engine="fbgemm")
    if qmodel is not None:
        model = qmodel
        print(f"[PTQ] Quantized model caricato: {qpath}")
    else:
        # 2) Float + ckpt + PTQ + save
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

    # Predizioni clean post-PTQ
    clean_by_batch = []
    with torch.inference_mode():
        for xb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            clean_by_batch.append(torch.argmax(logits, dim=1).cpu())

    clean_flat = torch.cat(clean_by_batch, dim=0) if len(clean_by_batch) else torch.tensor([], dtype=torch.long)
    num_classes = int(clean_flat.max().item()) + 1 if clean_flat.numel() > 0 else 2
    baseline_hist = np.bincount(clean_flat.numpy(), minlength=num_classes) if clean_flat.numel() > 0 else np.zeros(2, dtype=int)
    baseline_dist = baseline_hist / max(1, baseline_hist.sum())
    print(f"[BASELINE] pred dist = {baseline_dist}")

    return model, device, test_loader, clean_by_batch, baseline_hist, baseline_dist, num_classes


# ============================= Campagne =============================

def _evaluate_combo(model, device, test_loader, clean_by_batch, injector, combo, inj_id,
                    total_samples, baseline_hist, baseline_dist, num_classes):
    """
    Inietta una combo, valuta FRcrit + metriche di bias, ripristina.
    Ritorna (frcrit, faults, bias_dict, fault_hist, mism_by_clean, cnt_by_clean, cm_cf)
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


def run_statistical_srs_campaign(
    model, device, test_loader, clean_by_batch, all_faults, N,
    baseline_hist, baseline_dist, num_classes,
    pilot=None, eps=None, conf=None, block=None, budget_cap=None, seed=None,
    save_dir="results_minimal", dataset_name=None, net_name=None, bs=None, prefix_tag=""
):
    """
    Stima FRcrit_N con SRS i.i.d. su combinazioni:
    - pilot iniziale per stimare p_hat
    - calcolo n_target dalla normal approx
    - loop sequenziale con Wilson CI: stop quando halfwidth <= eps_ci o quando raggiungiamo budget_cap
    - mantiene Top-100 per FRcrit (con metriche di bias)
    - calcola un RIEPILOGO GLOBALE su tutte le iniezioni eseguite
    """
    # ----- Parametri statistici -----
    pilot   = pilot   if pilot   is not None else getattr(SETTINGS, "STAT_PILOT", 200)
    eps_ci  = eps     if eps     is not None else getattr(SETTINGS, "STAT_EPS",   0.005)
    conf    = conf    if conf    is not None else getattr(SETTINGS, "STAT_CONF",  0.95)
    block   = block   if block   is not None else getattr(SETTINGS, "STAT_BLOCK", 50)
    budget_cap = budget_cap if budget_cap is not None else getattr(SETTINGS, "STAT_BUDGET_CAP", None)
    seed    = seed    if seed    is not None else getattr(SETTINGS, "SEED", 0)

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

    # ----- Cartelle output -----
    dataset = dataset_name or SETTINGS.DATASET_NAME
    net     = net_name    or SETTINGS.NETWORK_NAME
    save_path = os.path.join(save_dir, dataset, net, f"batch_{bs}", "minimal_stat")
    os.makedirs(save_path, exist_ok=True)
    prefix = f"{dataset}_{net}_STAT_N{N}_batch{bs}{prefix_tag}"
    output_file = os.path.join(save_path, f"{prefix}.txt")

    # ----- PILOT -----
    gen = srs_combinations(all_faults, r=N, seed=seed, max_yield=pilot)
    sum_fr, n_inj = 0.0, 0
    top_heap = []  # (frcrit, inj_id, faults, bias)
    inj_id = 0

    pbar = tqdm(gen, total=pilot, desc=f"[STAT] pilot N={N}")
    for combo in pbar:
        inj_id += 1
        frcrit, faults, bias, fh, mbc, cbc, cm = _evaluate_combo(
            model, device, test_loader, clean_by_batch, injector, combo, inj_id, total_samples,
            baseline_hist, baseline_dist, num_classes
        )
        sum_fr += frcrit
        n_inj  += 1

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

    p_hat = (sum_fr / n_inj) if n_inj else 0.0
    n_target = choose_n_for_ci_normal(p_hat=max(p_hat, 1e-4), eps=eps_ci, conf=conf)
    if budget_cap:
        n_target = min(n_target, budget_cap)

    # ----- SEQUENZIALE con Wilson -----
    # consenti di superare n_target: il break avviene quando half <= eps_ci
    gen2 = srs_combinations(all_faults, r=N, seed=seed+1, max_yield=None)
    pbar2 = tqdm(total=None, desc=f"[STAT] N={N} target~{n_target} (eps_ci={eps_ci}, conf={conf})")


    block_acc = 0

    for combo in gen2:
        inj_id += 1
        frcrit, faults, bias, fh, mbc, cbc, cm = _evaluate_combo(
            model, device, test_loader, clean_by_batch, injector, combo, inj_id, total_samples,
            baseline_hist, baseline_dist, num_classes
        )
        sum_fr += frcrit
        n_inj  += 1
        block_acc += 1

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

        pbar2.update(1)
        if block_acc >= block:
            block_acc = 0
            p_curr = sum_fr / n_inj
            _, _, half_curr = wilson_ci(p_curr, n_inj, conf=conf)
            if half_curr <= eps_ci:
                break
            if budget_cap and n_inj >= budget_cap:
                break

    avg_frcrit = (sum_fr / n_inj) if n_inj else 0.0
    low, high, half = wilson_ci(avg_frcrit, n_inj, conf=conf)

    # ----- RIEPILOGO GLOBALE -----
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

    # BER aggregata (per classe clean)
    ber_per_class = []
    for c in range(num_classes):
        ber_c = (mism_by_clean_sum[c] / max(1, cnt_by_clean_sum[c]))
        ber_per_class.append(float(ber_c))
    BER = float(np.mean(ber_per_class)) if ber_per_class else 0.0

    # Agreement e asimmetria flip globali
    agree_global = float(np.trace(global_cm_cf)) / max(1, int(global_cm_cf.sum()))
    off_sum = int(global_cm_cf.sum() - np.trace(global_cm_cf))
    diff = np.abs(global_cm_cf - global_cm_cf.T)
    asym_num = int(diff[np.triu_indices(num_classes, k=1)].sum())
    flip_asym_global = float(asym_num) / max(1, off_sum)  # 0 = simmetrico

    # statistiche di frequenza bias per-injection
    maj_shares_arr = np.array(maj_shares) if maj_shares else np.array([])
    mean_share = float(maj_shares_arr.mean()) if maj_shares_arr.size else 0.0
    p90_share  = float(np.percentile(maj_shares_arr, 90)) if maj_shares_arr.size else 0.0
    frac_collapse_080 = float(np.mean(maj_shares_arr >= 0.80)) if maj_shares_arr.size else 0.0
    mean_kl = float(np.mean(kls)) if kls else 0.0

    # ----- SCRITTURA FILE -----
    top_sorted = sorted(top_heap, key=lambda x: x[0], reverse=True)
    with open(output_file, "w") as f:
        f.write(f"[STAT] N={N}  FRcrit_avg={avg_frcrit:.8f}  WilsonCI({int(conf*100)}%): [{low:.8f},{high:.8f}]  half={half:.8f}\n")
        f.write(
            f"injections_used={n_inj}  pilot={pilot}  eps_ci={eps_ci}  conf={conf}  block={block}  "
            f"budget_cap={budget_cap}  n_target~{n_target}\n"
        )
        f.write(
            f"[STAT] N={N}  FRcrit_avg={avg_frcrit:.8f}  "
            f"WilsonCI({int(conf*100)}%): [{low:.8f},{high:.8f}]  "
            f"eps_final={half:.8f}\n"
        )

        # E (opzionale) aggiungi eps_final anche alla riga seguente:
        # era: f"injections_used={n_inj}  pilot={pilot}  eps_ci={eps_ci}  conf={conf}  block={block}  budget_cap={budget_cap}  n_target~{n_target}\n"
        f.write(
            f"injections_used={n_inj}  pilot={pilot}  eps_ci={eps_ci}  eps_final={half:.8f}  "
            f"conf={conf}  block={block}  budget_cap={budget_cap}  n_target~{n_target}\n"
        )
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
        f.write(f"Top-{min(100, len(top_sorted))} worst injections (statistical SRS)\n")
        for rank, (frcrit, inj, faults, bias) in enumerate(top_sorted, 1):
            desc = ", ".join(f"{ft.layer_name}[{ft.tensor_index}] bit{ft.bits[0]}" for ft in faults)
            f.write(
                f"{rank:3d}) Inj {inj:6d} | FRcrit={frcrit:.6f} | "
                f"maj={bias['maj_cls']}@{bias['maj_share']:.2f} Δmax={bias['delta_max']:.3f} "
                f"KL={bias['kl']:.3f} | {desc}\n"
            )

    print(f"[STAT] N={N}  avgFRcrit={avg_frcrit:.6f}  Wilson±={half:.6f}  n={n_inj}  "
          f"(eps_ci={eps_ci}, conf={conf}, n_target~{n_target}) → {output_file}")
    return avg_frcrit, (low, high), n_inj, top_sorted, output_file


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
    exhaustive_up_to_n=3       # forzare ESAUSTIVA per N <= 3
):
    """
    Campagna 'classica':
      - ESAUSTIVA se (N <= exhaustive_up_to_n) oppure (C(num_faults, N) <= MAX_FAULTS),
      - altrimenti STATISTICA SRS con pilot + Wilson CI (stop sequenziale).
    In entrambi i casi salva Top-100 + metriche salienti.
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

        pbar = tqdm(fault_combos, total=max_iters, desc=f"N={N} (exa)")
        for inj_id, combo in enumerate(pbar, 1):
            frcrit, faults, bias, fh, mbc, cbc, cm = _evaluate_combo(
                model, device, test_loader, clean_by_batch, injector, combo, inj_id, total_samples,
                baseline_hist, baseline_dist, num_classes
            )
            sum_fr += frcrit
            n_injected += 1

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

        with open(output_file, "w") as f:
            f.write(f"Top-{min(TOP_K, len(top_sorted))} worst injections  (N={N})\n")
            f.write(f"Failure Rate (critical): {avg_frcrit:.8f}  on {n_injected} injections\n")
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
        # ===== STATISTICA SRS con CI =====
        # Nota: usiamo formattazione scientifica sicura per C(n,k)
        comb_str = _sci_format_comb(num_faults, N)
        print(f"[INFO] STATISTICA SRS: C({num_faults},{N})≈{comb_str} > MAX_FAULTS={MAX_FAULTS}. "
              f"Passo a stima con CI (pilot + Wilson stop).")
        avg_frcrit, ci, n_used, top_sorted, out_file = run_statistical_srs_campaign(
            model, device, test_loader, clean_by_batch, all_faults, N,
            baseline_hist, baseline_dist, num_classes,
            pilot=getattr(SETTINGS, "STAT_PILOT", 200),
            eps=getattr(SETTINGS, "STAT_EPS", 0.005),
            conf=getattr(SETTINGS, "STAT_CONF", 0.95),
            block=getattr(SETTINGS, "STAT_BLOCK", 50),
            budget_cap=getattr(SETTINGS, "STAT_BUDGET_CAP", None),
            seed=seed if seed is not None else getattr(SETTINGS, "SEED", 0),
            save_dir=save_dir, dataset_name=dataset, net_name=net_name, bs=bs
        )
        dt_min = (time.time() - t0) / 60.0
        if ci:
            low, high = ci[0], ci[1]
            print(f"[STAT] salvato {out_file} – {dt_min:.2f} min (avg FRcrit={avg_frcrit:.6f}, CI[{low:.6f},{high:.6f}], n={n_used})")
        else:
            print(f"[STAT] salvato {out_file} – {dt_min:.2f} min (avg FRcrit={avg_frcrit:.6f}, n={n_used})")
        return avg_frcrit, ci, n_used, top_sorted, out_file


# =================================== Main ===================================

if __name__ == "__main__":
    # Build + quantize + clean pass
    model, device, test_loader, clean_by_batch, baseline_hist, baseline_dist, num_classes = build_and_quantize_once()

    # Sweep: STAT con CI
    for N in [ 1, 2,3,4,5, 6, 7, 8, 9, 10, 50, 100, 150, 
             #384, 768, 960, 1104, 1408, 1728, 1984, 2048, 2208
             #576
             #384, 768, 960, 1104, 1408,
                 #1728, 1984, 2048, 2208 
                # 144, 200, 287, 288  
             ]:
              #1104, 800, 900, 1000, 1200, 1400, 1600, 1700, 1800, 2000, 2100, 2200
        run_fault_injection(
            model=model,
            device=device,
            test_loader=test_loader,
            clean_by_batch=clean_by_batch,
            baseline_hist=baseline_hist,
            baseline_dist=baseline_dist,
            num_classes=num_classes,
            N=N,
            MAX_FAULTS=0, #getattr(SETTINGS, "MAX_FAULTS", 1000),
            save_dir="results_minimal",
            seed=getattr(SETTINGS, "SEED", 0),
            exhaustive_up_to_n=-1  # esaustiva forzata fino a N
        )