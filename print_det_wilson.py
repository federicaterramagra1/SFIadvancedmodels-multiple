#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import random
import heapq
import csv
from itertools import islice, product, combinations

import numpy as np
import torch
from tqdm import tqdm

from faultManager.WeightFault import WeightFault
from faultManager.WeightFaultInjector import WeightFaultInjector
from utils import get_loader, load_from_dict, get_network
import SETTINGS


# ============================= Utils numerici & stampa sicura =============================

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


# ============================= Helper quantizzazione & fault =============================

def _get_quant_weight(module):
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
    z = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(conf, 1.96)
    p = min(max(p_hat, 1e-4), 1 - 1e-4)
    return math.ceil((z*z*p*(1-p))/(eps*eps))


# ============================= Build, quantize, clean pass =============================

def build_and_quantize_once():
    torch.backends.quantized.engine = "fbgemm"
    device = torch.device("cpu")

    model = get_network(SETTINGS.NETWORK_NAME, device, SETTINGS.DATASET_NAME)
    model.to(device).eval()
    ckpt = f"./trained_models/{SETTINGS.DATASET_NAME}_{SETTINGS.NETWORK_NAME}_trained.pth"
    if os.path.exists(ckpt):
        load_from_dict(model, device, ckpt)
        print("modello caricato")
    else:
        print(f"checkpoint non trovato: {ckpt} (proseguo senza)")

    train_loader, _, test_loader = get_loader(
        dataset_name=SETTINGS.DATASET_NAME,
        batch_size=SETTINGS.BATCH_SIZE,
        network_name=SETTINGS.NETWORK_NAME
    )

    if hasattr(model, "quantize_model"):
        model.to(device)
        model.quantize_model(calib_loader=train_loader)
        model.eval()
        print("quantizzazione completata")
    else:
        print("modello non quantizzabile (salto quantizzazione)")

    clean_by_batch = []
    with torch.inference_mode():
        for xb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            clean_by_batch.append(torch.argmax(logits, dim=1).cpu())

    clean_flat = torch.cat(clean_by_batch, dim=0)
    num_classes = int(clean_flat.max().item()) + 1 if clean_flat.numel() > 0 else 2
    baseline_hist = np.bincount(clean_flat.numpy(), minlength=num_classes)
    baseline_dist = baseline_hist / max(1, baseline_hist.sum())
    print(f"[BASELINE] pred dist = {baseline_dist}")

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

    all_faults = _build_all_faults(model, as_list=True)
    print(f"Siti single-bit totali: {len(all_faults)}")
    if len(all_faults) == 0:
        raise RuntimeError("Nessun fault enumerato: verifica quantizzazione e layer target.")

    return model, device, test_loader, clean_by_batch, baseline_hist, baseline_dist, num_classes


# ============================= Valutazione di UNA combinazione (riusabile) =============================

def _evaluate_combo(model, device, test_loader, clean_by_batch, injector, combo, inj_id,
                    total_samples, baseline_hist, baseline_dist, num_classes):
    faults = [WeightFault(injection=inj_id, layer_name=ln, tensor_index=ti, bits=[bt])
              for (ln, ti, bt) in combo]
    injected = False
    try:
        injector.inject_faults(faults, 'bit-flip')
        injected = True

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

                for c in range(num_classes):
                    msk = (clean_pred == c)
                    cnt_by_clean[c]  += int(msk.sum())
                    if msk.any():
                        mism_by_clean[c] += int((pred_f[msk] != c).sum())

        frcrit = mismatches / float(total_samples)

        fault_total = max(1, int(fault_hist.sum()))
        fault_dist = fault_hist / fault_total
        eps = 1e-12
        maj_cls = int(fault_hist.argmax()) if fault_hist.size > 0 else -1
        maj_share = float(fault_hist.max()) / fault_total if fault_hist.size > 0 else 0.0
        delta_max = float(np.max(np.abs(fault_dist - baseline_dist))) if fault_hist.size > 0 else 0.0
        kl = float(np.sum(fault_dist * np.log((fault_dist + eps) / (baseline_dist + eps)))) if fault_hist.size > 0 else 0.0

        off_sum = int(cm_cf.sum() - np.trace(cm_cf))
        diff = np.abs(cm_cf - cm_cf.T)
        asym_num = int(diff[np.triu_indices(num_classes, k=1)].sum()) if num_classes >= 2 else 0
        flip_asym = float(asym_num) / max(1, off_sum)
        agree = float(np.trace(cm_cf)) / max(1, int(cm_cf.sum()))

        bias = {"maj_cls": maj_cls, "maj_share": maj_share, "delta_max": delta_max,
                "kl": kl, "flip_asym": flip_asym, "agree": agree}

    finally:
        if injected:
            injector.restore_golden()

    return frcrit, faults, bias, fault_hist, mism_by_clean, cnt_by_clean, cm_cf


# ============================= Dump di DETTAGLIO per una specifica injection =============================

def _dump_single_injection_details(model, device, test_loader, clean_by_batch,
                                   faults, output_dir, save_prefix):
    """
    Re-inietta *esattamente* la lista 'faults' (WeightFault) e salva:
      - mismatches CSV (global_idx,pred_clean,pred_faulty)
      - TXT con confusion matrix, metriche, logit mean/std
      - dettaglio per-sito INT8/dequant prima/dopo
    """
    os.makedirs(output_dir, exist_ok=True)
    txt_path = os.path.join(output_dir, f"{save_prefix}_DETAIL.txt")
    csv_path = os.path.join(output_dir, f"{save_prefix}_mismatches.csv")

    # mapping name->module
    named = dict(model.named_modules())

    def _get_qwb(layer):
        if hasattr(layer, "weight"):
            try:
                return layer.weight(), layer.bias()
            except Exception:
                pass
        if hasattr(layer, "_packed_params") and hasattr(layer._packed_params, "_weight_bias"):
            return layer._packed_params._weight_bias()
        raise RuntimeError("Impossibile ottenere peso/bias quantizzati dal layer.")

    # fotografa "prima"
    before = []
    with torch.no_grad():
        for ft in faults:
            layer = named[ft.layer_name.replace('module.', '')]
            wq, _b = _get_qwb(layer)
            scale = float(wq.q_scale()); zp = int(wq.q_zero_point())
            i8 = int(wq.int_repr()[ft.tensor_index].item())
            u8 = (i8 + 256) % 256
            deq = float(wq.dequantize()[ft.tensor_index].item())
            before.append((ft.layer_name, ft.tensor_index, ft.bits[0], i8, u8, deq, scale, zp))

    injector = WeightFaultInjector(model)
    injected = False
    try:
        injector.inject_faults(faults, 'bit-flip')
        injected = True

        # after
        after = []
        with torch.no_grad():
            for ft in faults:
                layer = named[ft.layer_name.replace('module.', '')]
                wq, _b = _get_qwb(layer)
                scale = float(wq.q_scale()); zp = int(wq.q_zero_point())
                i8 = int(wq.int_repr()[ft.tensor_index].item())
                u8 = (i8 + 256) % 256
                deq = float(wq.dequantize()[ft.tensor_index].item())
                after.append((i8, u8, deq, scale, zp))

        # pass faulty + mismatch list
        total_samples = sum(len(t) for t in clean_by_batch)
        num_classes = int(torch.cat(clean_by_batch).max().item()) + 1
        baseline_hist = np.bincount(torch.cat(clean_by_batch).numpy(), minlength=num_classes)
        baseline_dist = baseline_hist / max(1, baseline_hist.sum())

        mismatches = 0
        fault_hist = np.zeros_like(baseline_hist, dtype=np.int64)
        cm_cf = np.zeros((num_classes, num_classes), dtype=np.int64)
        mismatch_list = []
        glb_off = 0
        faulty_logits_all = []

        with torch.inference_mode():
            for (batch_i, (xb, _yb)) in enumerate(test_loader):
                xb = xb.to(device)
                logits_f = model(xb)
                faulty_logits_all.append(logits_f.detach().cpu())
                pred_f = torch.argmax(logits_f, dim=1).cpu().numpy()
                np.add.at(fault_hist, pred_f, 1)
                clean_pred = clean_by_batch[batch_i].numpy()
                mism = (pred_f != clean_pred)
                mismatches += int(mism.sum())
                np.add.at(cm_cf, (clean_pred, pred_f), 1)
                if mism.any():
                    where = np.nonzero(mism)[0]
                    for j in where:
                        mismatch_list.append((int(glb_off + j), int(clean_pred[j]), int(pred_f[j])))
                glb_off += len(clean_pred)

        # clean logits (restore, poi pass)
        injector.restore_golden(); injected = False
        clean_logits_all = []
        with torch.inference_mode():
            for xb, _yb in test_loader:
                xb = xb.to(device)
                logits_c = model(xb)
                clean_logits_all.append(logits_c.detach().cpu())

        clean_logits_all  = torch.cat(clean_logits_all,  dim=0).numpy()
        faulty_logits_all = torch.cat(faulty_logits_all, dim=0).numpy()
        clean_mean = clean_logits_all.mean(axis=0); clean_std = clean_logits_all.std(axis=0)
        fault_mean = faulty_logits_all.mean(axis=0); fault_std = faulty_logits_all.std(axis=0)

        # metriche
        frcrit = mismatches / float(total_samples)
        fault_total = max(1, int(fault_hist.sum())); fault_dist = fault_hist / fault_total
        eps = 1e-12
        delta_max = float(np.max(np.abs(fault_dist - baseline_dist)))
        kl = float(np.sum(fault_dist * np.log((fault_dist + eps) / (baseline_dist + eps))))
        tv = 0.5 * float(np.sum(np.abs(fault_dist - baseline_dist)))
        H = lambda p: float(-np.sum(p * np.log(p + eps)))
        H_base  = H(baseline_dist); H_fault = H(fault_dist); dH = H_base - H_fault
        agree = float(np.trace(cm_cf)) / max(1, int(cm_cf.sum()))
        off_sum = int(cm_cf.sum() - np.trace(cm_cf))
        diff = np.abs(cm_cf - cm_cf.T); asym_num = int(diff[np.triu_indices(num_classes, k=1)].sum()) if off_sum>0 else 0
        flip_asym = float(asym_num) / max(1, off_sum) if off_sum>0 else 0.0
        cnt_by_clean  = cm_cf.sum(axis=1); correct_by_clean = np.diag(cm_cf)
        mism_by_clean = cnt_by_clean - correct_by_clean
        ber_per_class = (mism_by_clean / np.maximum(1, cnt_by_clean)).tolist()
        BER = float(np.mean(mism_by_clean / np.maximum(1, cnt_by_clean)))

        # --- salvataggi ---
        with open(csv_path, "w", newline="") as cf:
            wr = csv.writer(cf); wr.writerow(["global_idx","pred_clean","pred_faulty"]); wr.writerows(mismatch_list)

        with open(txt_path, "w") as f:
            f.write("=== DETTAGLIO injection peggiore ===\n")
            for i, ft in enumerate(faults):
                lname, t_idx, bit = ft.layer_name, ft.tensor_index, ft.bits[0]
                i8b,u8b,deqb,scb,zpb = before[i][3],before[i][4],before[i][5],before[i][6],before[i][7]
                i8a,u8a,deqa,sca,zpa = after[i]
                f.write(f"[{i+1:03d}] {lname}{t_idx} bit{bit}\n")
                f.write(f"     scale={scb:.6g} zp={zpb} | INT8 before={i8b:4d} (u8={u8b:3d}) deq_before={deqb:.7f}\n")
                f.write(f"                               -> INT8 after ={i8a:4d} (u8={u8a:3d}) deq_after ={deqa:.7f}\n")
            f.write("\n=== METRICHE ===\n")
            f.write(f"FRcrit={frcrit:.6f}\nBER={BER:.4f}\nper_class={ber_per_class}\n")
            f.write(f"Δmax={delta_max:.3f} KL={kl:.3f} TV={tv:.3f}\n")
            f.write(f"H_base={H_base:.3f} H_fault={H_fault:.3f} ΔH={dH:.3f}\n")
            f.write(f"agree={agree:.3f} flip_asym={flip_asym:.3f}\n\n")
            f.write("=== CONFUSION MATRIX clean→faulty ===\n")
            header = "      " + " ".join([f"f{c:>6d}" for c in range(len(cnt_by_clean))])
            f.write(header + "\n")
            for r in range(len(cnt_by_clean)):
                row_vals = " ".join([f"{int(cm_cf[r, c]):6d}" for c in range(len(cnt_by_clean))])
                f.write(f"c{r:>3d}  {row_vals}\n")
            f.write(f"row_sums={cnt_by_clean.tolist()}\n")
            f.write(f"diag={correct_by_clean.tolist()}\n\n")
            f.write("=== LOGIT mean/std ===\n")
            f.write(f"clean_mean={np.round(clean_mean,6).tolist()}\n")
            f.write(f"clean_std ={np.round(clean_std,6).tolist()}\n")
            f.write(f"fault_mean={np.round(fault_mean,6).tolist()}\n")
            f.write(f"fault_std ={np.round(fault_std,6).tolist()}\n")
        print(f"[SAVE] worst-injection detail TXT: {txt_path}")
        print(f"[SAVE] worst-injection mismatches CSV: {csv_path}")

    finally:
        if injected:
            injector.restore_golden()


# ============================= Campagna STATISTICA con Wilson =============================

def run_statistical_srs_campaign(
    model, device, test_loader, clean_by_batch, all_faults, N,
    baseline_hist, baseline_dist, num_classes,
    pilot=None, eps=None, conf=None, block=None, budget_cap=None, seed=None,
    save_dir="results_minimal", dataset_name=None, net_name=None, bs=None, prefix_tag=""
):
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

    # Aggregatori globali
    global_fault_hist = np.zeros_like(baseline_hist, dtype=np.int64)
    mism_by_clean_sum = np.zeros(num_classes, dtype=np.int64)
    cnt_by_clean_sum  = np.zeros(num_classes, dtype=np.int64)
    global_cm_cf      = np.zeros((num_classes, num_classes), dtype=np.int64)
    maj_shares, kls   = [], []

    # Output
    dataset = dataset_name or SETTINGS.DATASET_NAME
    net     = net_name    or SETTINGS.NETWORK_NAME
    save_path = os.path.join(save_dir, dataset, net, f"batch_{bs}", "minimal_stat")
    os.makedirs(save_path, exist_ok=True)
    prefix = f"{dataset}_{net}_STAT_N{N}_batch{bs}{prefix_tag}"
    output_file = os.path.join(save_path, f"{prefix}.txt")

    # PILOT
    gen = srs_combinations(all_faults, r=N, seed=seed, max_yield=pilot)
    sum_fr, n_inj = 0.0, 0
    top_heap = []  # (frcrit, inj_id, faults(list WeightFault-like tuples), bias)
    inj_id = 0

    pbar = tqdm(gen, total=pilot, desc=f"[STAT] pilot N={N}")
    for combo in pbar:
        inj_id += 1
        frcrit, faults, bias, fh, mbc, cbc, cm = _evaluate_combo(
            model, device, test_loader, clean_by_batch, injector, combo, inj_id, total_samples,
            baseline_hist, baseline_dist, num_classes
        )
        sum_fr += frcrit; n_inj += 1
        global_fault_hist += fh; mism_by_clean_sum += mbc; cnt_by_clean_sum += cbc; global_cm_cf += cm
        maj_shares.append(bias["maj_share"]); kls.append(bias["kl"])
        rec = (frcrit, inj_id, faults, bias)
        if len(top_heap) < 100:
            heapq.heappush(top_heap, rec)
        elif frcrit > top_heap[0][0]:
            heapq.heapreplace(top_heap, rec)

    p_hat = (sum_fr / n_inj) if n_inj else 0.0
    n_target = choose_n_for_ci_normal(p_hat=max(p_hat, 1e-4), eps=eps_ci, conf=conf)
    if budget_cap: n_target = min(n_target, budget_cap)
    _, _, half_pilot = wilson_ci(p_hat, n_inj, conf=conf)
    if half_pilot <= eps_ci:
        avg_frcrit = p_hat
        low, high, half = wilson_ci(avg_frcrit, n_inj, conf=conf)
        top_sorted = sorted(top_heap, key=lambda x: x[0], reverse=True)
    else:
        # SEQUENZIALE
        gen2 = srs_combinations(all_faults, r=N, seed=seed+1, max_yield=None)
        block_acc = 0
        pbar2 = tqdm(total=None, desc=f"[STAT] N={N} aiming ε={eps_ci} (conf={conf}, n_target~{n_target})")
        while True:
            combo = next(gen2)
            inj_id += 1
            frcrit, faults, bias, fh, mbc, cbc, cm = _evaluate_combo(
                model, device, test_loader, clean_by_batch, injector, combo, inj_id, total_samples,
                baseline_hist, baseline_dist, num_classes
            )
            sum_fr += frcrit; n_inj += 1; block_acc += 1
            global_fault_hist += fh; mism_by_clean_sum += mbc; cnt_by_clean_sum += cbc; global_cm_cf += cm
            maj_shares.append(bias["maj_share"]); kls.append(bias["kl"])
            rec = (frcrit, inj_id, faults, bias)
            if len(top_heap) < 100:
                heapq.heappush(top_heap, rec)
            elif frcrit > top_heap[0][0]:
                heapq.heapreplace(top_heap, rec)
            pbar2.update(1)
            if (block_acc >= block) or (n_inj % block == 0):
                block_acc = 0
                p_curr = sum_fr / n_inj
                _, _, half_curr = wilson_ci(p_curr, n_inj, conf=conf)
                if half_curr <= eps_ci:
                    break
                if budget_cap and n_inj >= budget_cap:
                    print(f"[STAT] budget_cap raggiunto: n={n_inj}, half={half_curr:.6f} (> ε={eps_ci})")
                    break
        avg_frcrit = (sum_fr / n_inj) if n_inj else 0.0
        low, high, half = wilson_ci(avg_frcrit, n_inj, conf=conf)
        top_sorted = sorted(top_heap, key=lambda x: x[0], reverse=True)

    # RIEPILOGO GLOBALE
    total_preds = int(global_fault_hist.sum())
    global_fault_dist = (global_fault_hist / total_preds) if total_preds > 0 else baseline_dist
    tiny = 1e-12
    global_delta_max = float(np.max(np.abs(global_fault_dist - baseline_dist)))
    global_kl = float(np.sum(global_fault_dist * np.log((global_fault_dist + tiny)/(baseline_dist + tiny))))
    global_tv = 0.5 * float(np.sum(np.abs(global_fault_dist - baseline_dist)))
    H = lambda p: float(-np.sum(p * np.log(p + tiny)))
    entropy_baseline = H(baseline_dist); entropy_global = H(global_fault_dist); entropy_drop = entropy_baseline - entropy_global
    ber_per_class = []
    for c in range(num_classes):
        ber_c = (mism_by_clean_sum[c] / max(1, cnt_by_clean_sum[c]))
        ber_per_class.append(float(ber_c))
    BER = float(np.mean(ber_per_class)) if ber_per_class else 0.0
    agree_global = float(np.trace(global_cm_cf)) / max(1, int(global_cm_cf.sum()))
    off_sum = int(global_cm_cf.sum() - np.trace(global_cm_cf))
    diff = np.abs(global_cm_cf - global_cm_cf.T)
    asym_num = int(diff[np.triu_indices(num_classes, k=1)].sum()) if off_sum>0 else 0
    flip_asym_global = float(asym_num) / max(1, off_sum) if off_sum>0 else 0.0
    maj_shares_arr = np.array(maj_shares) if maj_shares else np.array([])
    mean_share = float(maj_shares_arr.mean()) if maj_shares_arr.size else 0.0
    p90_share  = float(np.percentile(maj_shares_arr, 90)) if maj_shares_arr.size else 0.0
    frac_collapse_080 = float(np.mean(maj_shares_arr >= 0.80)) if maj_shares_arr.size else 0.0
    mean_kl = float(np.mean(kls)) if kls else 0.0

    # SCRITTURA FILE GLOBALE
    with open(output_file, "w") as f:
        f.write(f"[STAT] N={N}  FRcrit_avg={avg_frcrit:.8f}  WilsonCI({int(conf*100)}%): [{low:.8f},{high:.8f}]  half={half:.8f}\n")
        f.write(f"injections_used={n_inj}  pilot={pilot}  eps_ci={eps_ci}  conf={conf}  block={block}  budget_cap={budget_cap}\n")
        f.write(f"baseline_pred_dist={baseline_dist.tolist()}\n")
        f.write(
            "global_summary_over_injections: "
            f"fault_pred_dist={global_fault_dist.tolist()} Δmax={global_delta_max:.3f} KL={global_kl:.3f} TV={global_tv:.3f} "
            f"H_baseline={entropy_baseline:.3f} H_global={entropy_global:.3f} ΔH={entropy_drop:.3f} "
            f"BER={BER:.4f} per_class={ber_per_class} agree={agree_global:.3f} flip_asym={flip_asym_global:.3f}\n\n"
        )
        # Confusion matrix aggregata
        f.write("=== CONFUSION MATRIX (aggregata su tutte le injection) clean→faulty ===\n")
        header = "      " + " ".join([f"f{c:>6d}" for c in range(num_classes)])
        f.write(header + "\n")
        for r in range(num_classes):
            row_vals = " ".join([f"{int(global_cm_cf[r, c]):6d}" for c in range(num_classes)])
            f.write(f"c{r:>3d}  {row_vals}\n")
        f.write(f"row_sums={(global_cm_cf.sum(axis=1)).tolist()}\n")
        f.write(f"diag={(np.diag(global_cm_cf)).tolist()}\n\n")

        f.write(f"bias_frequency: mean_share={mean_share:.3f} p90_share={p90_share:.3f} mean_KL={mean_kl:.3f} "
                f"frac_collapse(share≥0.80)={frac_collapse_080:.3f}\n\n")

        f.write(f"Top-{min(100, len(top_sorted))} worst injections (statistical SRS)\n")
        for rank, (frcrit, inj, faults, bias) in enumerate(top_sorted, 1):
            desc = ", ".join(f"{ft.layer_name}[{ft.tensor_index}] bit{ft.bits[0]}" for ft in faults)
            f.write(
                f"{rank:3d}) Inj {inj:6d} | FRcrit={frcrit:.6f} | "
                f"maj={bias['maj_cls']}@{bias['maj_share']:.2f} Δmax={bias['delta_max']:.3f} "
                f"KL={bias['kl']:.3f} | {desc}\n"
            )

    print(f"[STAT] N={N}  avgFRcrit={avg_frcrit:.6f}  Wilson±={half:.6f}  n={n_inj} → {output_file}")

    # RIESAME injection peggiore (stesso formato della parte "esaustiva")
    if top_sorted:
        worst = top_sorted[0]
        _, worst_inj_id, worst_faults, _ = worst
        # I 'faults' di top_sorted sono oggetti WeightFault creati in _evaluate_combo → riusabili
        # ma per robustezza ricreo la lista (layer_name,tensor_index,bit)
        faults_rebuilt = []
        for ft in worst_faults:
            faults_rebuilt.append(WeightFault(injection=worst_inj_id,
                                              layer_name=ft.layer_name,
                                              tensor_index=ft.tensor_index,
                                              bits=list(ft.bits)))
        # dump dettagli
        detail_prefix = os.path.splitext(os.path.basename(output_file))[0] + "_WORST"
        _dump_single_injection_details(model, device, test_loader, clean_by_batch,
                                       faults_rebuilt, output_dir=os.path.dirname(output_file),
                                       save_prefix=detail_prefix)

    return avg_frcrit, (low, high), n_inj, top_sorted, output_file


# ============================= Campagna "dispatcher" =============================

def run_fault_injection(
    model, device, test_loader, clean_by_batch,
    baseline_hist, baseline_dist, num_classes,
    N, MAX_FAULTS=4_000_000, save_dir="results_minimal", seed=None, exhaustive_up_to_n=3
):
    t0 = time.time()
    dataset = SETTINGS.DATASET_NAME
    net_name = SETTINGS.NETWORK_NAME
    bs = getattr(test_loader, "batch_size", None) or SETTINGS.BATCH_SIZE
    save_path = os.path.join(save_dir, dataset, net_name, f"batch_{bs}", "minimal")
    os.makedirs(save_path, exist_ok=True)

    all_faults = _build_all_faults(model, as_list=True)
    num_faults = len(all_faults)
    if num_faults == 0:
        raise RuntimeError("Lista fault vuota: impossibile proseguire.")
    if N > num_faults:
        print(f"[WARN] N={N} > num_faults={num_faults}. Ridimensiono N a {num_faults}.")
        N = num_faults

    force_exhaustive = (N <= exhaustive_up_to_n)
    total_possible = math.comb(num_faults, N)

    if force_exhaustive or total_possible <= MAX_FAULTS:
        print(f"[INFO] ESAUSTIVA (N={N}, combinazioni={_sci_format_comb(num_faults, N)})...")
        # (qui potresti incollare la tua branch ESA già pronta, se mai servisse)
        raise RuntimeError("Per questa esecuzione vogliamo solo il ramo STATISTICO. Imposta MAX_FAULTS=0 e exhaustive_up_to_n=-1.")
    else:
        comb_str = _sci_format_comb(num_faults, N)
        print(f"[INFO] STATISTICA SRS: C({num_faults},{N})≈{comb_str} > MAX_FAULTS={MAX_FAULTS}. Stima con Wilson.")
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
        low, high = ci if ci else (None, None)
        print(f"[STAT] salvato {out_file} – {dt_min:.2f} min (avg FRcrit={avg_frcrit:.6f}, CI[{low:.6f},{high:.6f}], n={n_used})")
        return avg_frcrit, ci, n_used, top_sorted, out_file


# =================================== Main ===================================

if __name__ == "__main__":
    # Build + quantize + clean pass
    model, device, test_loader, clean_by_batch, baseline_hist, baseline_dist, num_classes = build_and_quantize_once()

    # === LANCIO STATISTICO con WILSON per K (N) = 144 ===
    K = 384
    avg, ci, n_used, _, out_file = run_fault_injection(
        model=model,
        device=device,
        test_loader=test_loader,
        clean_by_batch=clean_by_batch,
        baseline_hist=baseline_hist,
        baseline_dist=baseline_dist,
        num_classes=num_classes,
        N=K,
        MAX_FAULTS=0,          # forza ramo statistico
        save_dir="results_minimal",
        seed=getattr(SETTINGS, "SEED", 0),
        exhaustive_up_to_n=-1  # disattiva ESA
    )
    low, high = ci if ci else (None, None)
    print(f"[WILSON] K={K} | FR_avg={avg:.6g} | CI=[{low:.6g},{high:.6g}] | injections={n_used} | file={out_file}")
