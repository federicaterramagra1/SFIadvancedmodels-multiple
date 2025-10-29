#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import math
import random
import heapq
from itertools import product, combinations
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from faultManager.WeightFault import WeightFault
from faultManager.WeightFaultInjector import WeightFaultInjector
import SETTINGS
from utils import get_loader, load_from_dict, get_network, load_quantized_model, save_quantized_model
from adapters.builders import build_resnet20_cifar10_pretrained_int8_and_clean
from adapters.resnet20_cifar10_pretrained_int8 import get_resnet20_cifar10_int8

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

def _unpack_half(h):
    if isinstance(h, (tuple, list)) and h:
        return float(h[0])
    if isinstance(h, (int, float, np.floating)):
        return float(h)
    return float("nan")

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

def _initial_n_infinite(e_goal: float, conf: float, p0: float = 0.5) -> int:
    z = _z_from_conf(conf)
    return int(math.ceil((z*z*p0*(1.0-p0)) / (e_goal*e_goal)))

def decide_sampling_policy(num_faults: int, K: int, e_goal: float, conf: float, p0: float = 0.5,
                           exhaustive_cap: Optional[int] = None):
    """
    Ritorna: (N_pop, use_fpc, force_exhaustive, n_inf, n_fpc_or_None, ratio)
    """
    N_pop = math.comb(num_faults, K)
    if N_pop <= 0:
        return 0, False, False, 0, None, 0.0

    n_inf = _initial_n_infinite(e_goal, conf, p0)
    ratio = n_inf / N_pop

    # Regola 5% STRETTAMENTE maggiore
    use_fpc = (ratio > 0.05)
    n_fpc = None
    if use_fpc:
        # usa p0 per il planning iniziale; l'iterativo poi aggiorna con p_hat
        n_fpc = plan_n_with_fpc(p_hat=p0, e_target=e_goal, conf=conf, N_pop=N_pop)

    # forza esaustiva se la n_FPC copre sostanzialmente tutta la popolazione
    force_exhaustive = False
    if use_fpc and n_fpc is not None:
        if n_fpc >= 0.95 * N_pop:
            force_exhaustive = True

    # opzionale: anche un 'tetto' assoluto a N_pop per andare esaustivo
    if exhaustive_cap is not None and N_pop <= exhaustive_cap:
        force_exhaustive = True

    return N_pop, use_fpc, force_exhaustive, n_inf, n_fpc, ratio


# ============================= Sampler & statistica =============================

def srs_combinations(pool, r, seed=None, max_yield=None):
    rnd = random.Random(seed)
    n = len(pool)
    if n == 0:
        return
    r = min(r, n)
    produced = 0
    while max_yield is None or produced < max_yield:
        idxs = rnd.sample(range(n), r)
        yield tuple(pool[i] for i in idxs)
        produced += 1

def _z_from_conf(conf: float) -> float:
    table = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576, 0.999: 3.291}
    return table.get(conf, 1.96)

def fpc_factor(N_pop: Optional[int], n: int) -> float:
    if not N_pop or N_pop <= 0:
        return 1.0
    if n <= 1 or n >= N_pop:
        val = max(0.0, (N_pop - n) / max(1, N_pop - 1))
        return math.sqrt(val)
    return math.sqrt((N_pop - n) / (N_pop - 1))

def halfwidth_normal_fpc(p_hat: float, n: int, conf: float, N_pop: Optional[int]) -> float:
    z = _z_from_conf(conf)
    p = min(max(p_hat, 1e-12), 1.0 - 1e-12)
    hw = z * math.sqrt(p * (1.0 - p) / max(1, n))
    return hw * fpc_factor(N_pop, n)

def plan_n_with_fpc(p_hat: float, e_target: float, conf: float, N_pop: Optional[int]) -> int:
    z = _z_from_conf(conf)
    p = min(max(p_hat, 1e-6), 1.0 - 1e-6)
    if not N_pop or N_pop <= 0:
        n = math.ceil((z * z * p * (1.0 - p)) / (e_target * e_target))
        return max(1, n)
    denom = 1.0 + (e_target * e_target * (N_pop - 1.0)) / (z * z * p * (1.0 - p))
    n = N_pop / denom
    return int(min(N_pop, max(1, math.ceil(n))))

def ep_next_error(e_goal: float, E_hat: float, p_hat: float) -> float:
    if E_hat <= 0.0:
        return e_goal
    thresh = E_hat / 3.0
    if thresh <= e_goal:
        return e_goal
    k = 4.0 * (thresh - e_goal)
    val = -k * (p_hat ** 2) + k * p_hat + e_goal
    return float(min(max(val, e_goal), thresh))

def wald_ci(p_hat: float, n: int, conf: float, N_pop: Optional[int]):
    half = halfwidth_normal_fpc(p_hat, n, conf, N_pop)
    z = _z_from_conf(conf)
    se = half / max(z, 1e-12)
    low = max(0.0, p_hat - half)
    high = min(1.0, p_hat + half)
    return low, high, half, se

# ---------- Wilson (plain) ----------
def wilson_ci(p_hat: float, n: int, conf: float = 0.95):
    if n <= 0:
        return 0.0, 1.0, 0.5
    z = _z_from_conf(conf)
    p = min(max(p_hat, 1e-12), 1.0 - 1e-12)
    z2 = z * z
    denom = 1.0 + z2 / n
    half = (z * math.sqrt((p * (1.0 - p) / n) + (z2 / (4.0 * n * n)))) / denom
    center = (p + z2 / (2.0 * n)) / denom
    low = max(0.0, center - half)
    high = min(1.0, center + half)
    return low, high, half

def wilson_halfwidth(p_hat: float, n: int, conf: float = 0.95) -> float:
    return wilson_ci(p_hat, n, conf)[2]

# ---------- Wilson con FPC ----------
def wilson_ci_fpc(p_hat: float, n: int, conf: float = 0.95, N_pop: Optional[int] = None):
    """
    Wilson score interval con FPC:
    - se N_pop è dato e n >= N_pop: censimento ⇒ intervallo degen. [p̂, p̂], half=0
    - altrimenti applica FPC^2 al termine di varianza p(1-p)/n
    """
    if n <= 0:
        return 0.0, 1.0, 0.5
    if N_pop and n >= N_pop:
        return p_hat, p_hat, 0.0

    z = _z_from_conf(conf)
    p = min(max(p_hat, 1.0e-12), 1.0 - 1.0e-12)
    z2 = z * z

    # FPC^2 sul termine di varianza
    if N_pop and (N_pop > 1) and (n < N_pop):
        f2 = (N_pop - n) / (N_pop - 1.0)
    else:
        f2 = 1.0

    denom = 1.0 + z2 / n
    var = p * (1.0 - p) / n
    var *= f2

    center = (p + z2 / (2.0 * n)) / denom
    half = (z * math.sqrt(var + (z2 / (4.0 * n * n)))) / denom

    low = max(0.0, center - half)
    high = min(1.0, center + half)
    return low, high, half

def wilson_halfwidth_fpc(p_hat: float, n: int, conf: float = 0.95, N_pop: Optional[int] = None) -> float:
    return wilson_ci_fpc(p_hat, n, conf, N_pop)[2]

def _plan_n_wilson(p_hat: float, e_target: float, conf: float,
                   start_n: int = 1, cap: Optional[int] = None, N_pop: Optional[int] = None) -> int:
    """
    Pianifica n con bisezione, usando halfwidth Wilson; se N_pop è dato, usa Wilson-FPC.
    """
    # clamp di sicurezza
    p = min(max(p_hat, 1e-6), 1 - 1e-6)

    z = _z_from_conf(conf)
    # upper bound iniziale (grezzo) con formula normale; poi bisezione con Wilson(_FPC)
    n_wald = max(1, int(math.ceil((z * z * p * (1 - p)) / (e_target * e_target))))
    ub = max(n_wald * 4, start_n + 1)
    if N_pop:
        ub = min(ub, int(N_pop))
    if cap:
        ub = min(ub, cap)

    lo = max(1, start_n)
    hi = max(lo + 1, ub)

    # selettore halfwidth
    hw_fn = (lambda pp, nn: wilson_halfwidth_fpc(pp, nn, conf, N_pop)) if N_pop else (lambda pp, nn: wilson_halfwidth(pp, nn, conf))

    while lo < hi:
        mid = (lo + hi) // 2
        if hw_fn(p, mid) <= e_target:
            hi = mid
        else:
            lo = mid + 1
    return lo

# ============================= Build, quantize, clean pass =============================

def build_and_quantize_once():
    torch.backends.quantized.engine = "fbgemm"
    device = torch.device("cpu")

    ds_name = getattr(SETTINGS, "DATASET_NAME", getattr(SETTINGS, "DATASET", ""))
    net_name = getattr(SETTINGS, "NETWORK_NAME", getattr(SETTINGS, "NETWORK", ""))

    train_loader, _, test_loader = get_loader(
        dataset_name=ds_name,
        batch_size=getattr(SETTINGS, "BATCH_SIZE", 64),
        network_name=net_name,
    )

    qmodel, qpath = load_quantized_model(ds_name, net_name, device="cpu", engine="fbgemm")
    if qmodel is not None:
        model = qmodel
        print(f"[PTQ] Quantized model caricato: {qpath}")
    else:
        model = get_network(net_name, device, ds_name)
        model.to(device).eval()

        ckpt = f"./trained_models/{ds_name}_{net_name}_trained.pth"
        if os.path.exists(ckpt):
            load_from_dict(model, device, ckpt)
            print("[CKPT] modello float caricato")
        else:
            print(f"[WARN] checkpoint float non trovato: {ckpt} (proseguo senza)")

        if hasattr(model, "quantize_model"):
            model.quantize_model(calib_loader=train_loader)
            model.eval()
            qsave = save_quantized_model(model, ds_name, net_name, engine="fbgemm")
            print(f"[PTQ] quantizzazione completata (CPU) e salvata: {qsave}")
        else:
            print("[PTQ] modello non quantizzabile – salto")

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
    print(f"[BASELINE] pred dist = {baseline_dist.tolist()}")

    return model, device, test_loader, clean_by_batch, baseline_hist, baseline_dist, num_classes

# ============================= Valutazione singola combo =============================

def _evaluate_combo(model, device, test_loader, clean_by_batch, injector, combo, inj_id,
                    total_samples, baseline_hist, baseline_dist, num_classes):
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

                for c in range(num_classes):
                    msk = (clean_pred == c)
                    cnt_by_clean[c]  += int(msk.sum())
                    if msk.any():
                        mism_by_clean[c] += int((pred_f[msk] != c).sum())

        frcrit = mismatches / float(total_samples)

        fault_total = max(1, int(fault_hist.sum()))
        fault_dist = fault_hist / fault_total
        maj_cls = int(fault_hist.argmax()) if fault_hist.size > 0 else -1
        maj_share = float(fault_hist.max()) / fault_total if fault_hist.size > 0 else 0.0
        delta_max = float(np.max(np.abs(fault_dist - baseline_dist))) if fault_hist.size > 0 else 0.0
        eps = 1e-12
        kl = float(np.sum(fault_dist * np.log((fault_dist + eps) / (baseline_dist + eps)))) if fault_hist.size > 0 else 0.0

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
            "agree": agree
        }

    finally:
        injector.restore_golden()

    return frcrit, faults, bias, fault_hist, mism_by_clean, cnt_by_clean, cm_cf

# ============================= Campagna PROPOSED (iterativa) =============================

def run_statistical_iterative_ep_fpc_campaign(
    model, device, test_loader, clean_by_batch, all_faults, N,
    baseline_hist, baseline_dist, num_classes,
    pilot=None, e_goal=None, conf=None, block=None, budget_cap=None, seed=None,
    use_fpc: bool = False, N_pop: Optional[int] = None,
    save_dir="results_minimal", dataset_name=None, net_name=None, bs=None,
    ep_control: str = "wilson",
    prefix_tag="_EP"
):
    """
    Iterative EP campaign with optional FPC.
    - Se FPC è ON: WOR globale (senza rimpiazzo tra draw) per QUALSIASI K; cap n <= N_pop.
    - Se FPC è OFF: WR (con rimpiazzo tra draw), come sempre.
    """
    # Defaults
    pilot   = pilot   if pilot   is not None else getattr(SETTINGS, "STAT_PILOT", 200)
    e_goal  = e_goal  if e_goal  is not None else getattr(SETTINGS, "STAT_EPS",   0.005)
    conf    = conf    if conf    is not None else getattr(SETTINGS, "STAT_CONF",  0.95)
    block   = block   if block   is not None else getattr(SETTINGS, "STAT_BLOCK", 50)
    budget_cap = budget_cap if budget_cap is not None else getattr(SETTINGS, "STAT_BUDGET_CAP", None)
    seed    = seed    if seed    is not None else getattr(SETTINGS, "SEED", 0)

    # Injector & basic checks
    injector = WeightFaultInjector(model)
    total_samples = sum(len(t) for t in clean_by_batch)
    if total_samples == 0:
        raise RuntimeError("Empty test loader.")

    # Output paths
    dataset = dataset_name or SETTINGS.DATASET_NAME
    net     = net_name    or SETTINGS.NETWORK_NAME
    bs = getattr(test_loader, "batch_size", None) or SETTINGS.BATCH_SIZE
    save_path = os.path.join(save_dir, dataset, net, f"batch_{bs}", "minimal_iterEP")
    os.makedirs(save_path, exist_ok=True)
    prefix = f"{dataset}_{net}_PROP_K{N}_batch{bs}{prefix_tag}{('_' + ep_control) if ep_control else ''}"
    output_file = os.path.join(save_path, f"{prefix}.txt")

    # Global accumulators
    global_fault_hist = np.zeros_like(baseline_hist, dtype=np.int64)
    mism_by_clean_sum = np.zeros(num_classes, dtype=np.int64)
    cnt_by_clean_sum  = np.zeros(num_classes, dtype=np.int64)
    global_cm_cf      = np.zeros((num_classes, num_classes), dtype=np.int64)
    maj_shares, kls   = [], []

    mean_frcrit = 0.0
    m2_frcrit   = 0.0
    count_fr    = 0
    top_heap    = []  # min-heap for top-100 worst injections

    # --- WOR/WR design ---
    use_wor = bool(use_fpc and N_pop and N >= 1)
    rng = random.Random(seed)

    # WOR K=1: permutazione dei singoli fault
    k1_order = None
    k1_ptr = 0
    if use_wor and N == 1:
        k1_order = list(range(len(all_faults)))
        rng.shuffle(k1_order)

    # WOR K>1: set dei combo già visti (chiavi canoniche ordinate)
    seen = set() if (use_wor and N > 1) else None

    def _combo_key(combo):
        return tuple(sorted(combo))

    def _next_wor_combo():
        """Ritorna un combo unico (WOR) oppure None se popolazione esaurita."""
        nonlocal k1_ptr
        if N == 1:
            if k1_order is None or k1_ptr >= len(k1_order):
                return None
            idx = k1_order[k1_ptr]
            k1_ptr += 1
            return (all_faults[idx],)
        else:
            if seen is not None and len(seen) >= int(N_pop):
                return None
            # genera combo casuali finché non trovi uno nuovo
            while True:
                combo = tuple(rng.sample(all_faults, N))  # senza ripetizioni intra-combo
                key = _combo_key(combo)
                if key not in seen:
                    seen.add(key)
                    return combo
                # con n << N_pop, collisioni rarissime

    # Generators WR (vecchio comportamento) quando FPC=OFF
    if not use_wor:
        gen = srs_combinations(all_faults, r=N, seed=seed,   max_yield=pilot)
        gen_more = srs_combinations(all_faults, r=N, seed=seed+1, max_yield=None)
    else:
        gen = None
        gen_more = None

    # Clamp pilot se WOR e abbiamo N_pop
    if use_wor and N_pop:
        pilot = min(pilot, int(N_pop))
        if N == 1:
            pilot = min(pilot, len(all_faults))

    # ------------------ PILOT ------------------
    sum_fr, n_inj, inj_id = 0.0, 0, 0
    if use_wor:
        pbar = tqdm(total=pilot, desc=f"[PROP] pilot K={N} (WOR)")
        injected = 0
        while injected < pilot:
            combo = _next_wor_combo()
            if combo is None:
                break  # popolazione esaurita
            inj_id += 1
            frcrit, faults, bias, fh, mbc, cbc, cm = _evaluate_combo(
                model, device, test_loader, clean_by_batch, injector, combo, inj_id, total_samples,
                baseline_hist, baseline_dist, num_classes
            )
            sum_fr += frcrit
            n_inj  += 1
            injected += 1
            pbar.update(1)

            count_fr += 1
            delta = frcrit - mean_frcrit
            mean_frcrit += delta / count_fr
            m2_frcrit += delta * (frcrit - mean_frcrit)

            global_fault_hist += fh
            mism_by_clean_sum += mbc
            cnt_by_clean_sum  += cbc
            global_cm_cf      += cm
            maj_shares.append(bias["maj_share"])
            kls.append(bias["kl"])

            if len(top_heap) < 100:
                heapq.heappush(top_heap, (frcrit, inj_id, faults, bias))
            elif frcrit > top_heap[0][0]:
                heapq.heapreplace(top_heap, (frcrit, inj_id, faults, bias))
    else:
        pbar = tqdm(gen, total=pilot, desc=f"[PROP] pilot K={N} (WR)")
        for combo in pbar:
            inj_id += 1
            frcrit, faults, bias, fh, mbc, cbc, cm = _evaluate_combo(
                model, device, test_loader, clean_by_batch, injector, combo, inj_id, total_samples,
                baseline_hist, baseline_dist, num_classes
            )
            sum_fr += frcrit
            n_inj  += 1

            count_fr += 1
            delta = frcrit - mean_frcrit
            mean_frcrit += delta / count_fr
            m2_frcrit += delta * (frcrit - mean_frcrit)

            global_fault_hist += fh
            mism_by_clean_sum += mbc
            cnt_by_clean_sum  += cbc
            global_cm_cf      += cm
            maj_shares.append(bias["maj_share"])
            kls.append(bias["kl"])

            if len(top_heap) < 100:
                heapq.heappush(top_heap, (frcrit, inj_id, faults, bias))
            elif frcrit > top_heap[0][0]:
                heapq.heapreplace(top_heap, (frcrit, inj_id, faults, bias))

    # Stima iniziale
    p_hat = (sum_fr / n_inj) if n_inj else 0.0
    E_wald = halfwidth_normal_fpc(p_hat, n_inj, conf, (N_pop if use_fpc else None))
    E_wil  = (wilson_halfwidth_fpc(p_hat, n_inj, conf, N_pop) if use_fpc else wilson_halfwidth(p_hat, n_inj, conf))
    E_seq = [(n_inj, E_wald, E_wil)]
    E_hat = E_wald if ep_control == "wald" else E_wil
    print(f"[PROP] after pilot: p̂={p_hat:.6f}  E_wald={E_wald:.6f}  E_wilson={E_wil:.6f}  (ctrl={ep_control})")

    # ------------------ ITERATIVE STEPS ------------------
    steps = 0
    while True:
        steps += 1
        if E_hat <= e_goal:
            break
        if budget_cap and n_inj >= budget_cap:
            break

        # EP controller: prossimo target di errore
        e_next = ep_next_error(e_goal=e_goal, E_hat=E_hat, p_hat=p_hat)

        # Pianifica n totale
        if ep_control == "wald":
            n_tot_desired = plan_n_with_fpc(
                p_hat=p_hat, e_target=e_next, conf=conf,
                N_pop=(N_pop if use_fpc else None)
            )
        else:
            n_tot_desired = _plan_n_wilson(
                p_hat=p_hat, e_target=e_next, conf=conf,
                start_n=n_inj, cap=budget_cap, N_pop=(N_pop if use_fpc else None)
            )

        # Cap FPC: n <= N_pop
        if use_fpc and N_pop:
            n_tot_desired = min(n_tot_desired, int(N_pop))

        add_needed = max(0, n_tot_desired - n_inj)
        if budget_cap:
            add_needed = min(add_needed, max(0, budget_cap - n_inj))
        if add_needed <= 0:
            break

        pbar2 = tqdm(total=add_needed, desc=f"[PROP/{ep_control}] step#{steps} to n={n_tot_desired} (e_next={e_next:.6g})")
        to_do = add_needed
        while to_do > 0:
            if use_wor:
                combo = _next_wor_combo()
                if combo is None:
                    # popolazione esaurita: non posso crescere oltre
                    to_do = 0
                    break
            else:
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

            count_fr += 1
            delta = frcrit - mean_frcrit
            mean_frcrit += delta / count_fr
            m2_frcrit += delta * (frcrit - mean_frcrit)

            global_fault_hist += fh
            mism_by_clean_sum += mbc
            cnt_by_clean_sum  += cbc
            global_cm_cf      += cm
            maj_shares.append(bias["maj_share"])
            kls.append(bias["kl"])

            if len(top_heap) < 100:
                heapq.heappush(top_heap, (frcrit, inj_id, faults, bias))
            elif frcrit > top_heap[0][0]:
                heapq.heapreplace(top_heap, (frcrit, inj_id, faults, bias))

            if (n_inj % block) == 0:
                p_hat = (sum_fr / n_inj)
                E_wald = halfwidth_normal_fpc(p_hat, n_inj, conf, (N_pop if use_fpc else None))
                E_wil  = (wilson_halfwidth_fpc(p_hat, n_inj, conf, N_pop) if use_fpc else wilson_halfwidth(p_hat, n_inj, conf))
                E_hat  = E_wald if ep_control == "wald" else E_wil
                E_seq.append((n_inj, E_wald, E_wil))
                pbar2.set_postfix_str(f"p̂={p_hat:.6f} Ew={E_wald:.6f} Ei={E_wil:.6f}")
                if E_hat <= e_goal or (budget_cap and n_inj >= budget_cap):
                    break

        # Update dopo il batch
        p_hat = (sum_fr / n_inj)
        E_wald = halfwidth_normal_fpc(p_hat, n_inj, conf, (N_pop if use_fpc else None))
        E_wil  = (wilson_halfwidth_fpc(p_hat, n_inj, conf, N_pop) if use_fpc else wilson_halfwidth(p_hat, n_inj, conf))
        E_hat  = E_wald if ep_control == "wald" else E_wil
        E_seq.append((n_inj, E_wald, E_wil))
        print(f"[PROP/{ep_control}] update: n={n_inj} p̂={p_hat:.6f}  Ew={E_wald:.6f}  Ei={E_wil:.6f}")

        if E_hat <= e_goal or (budget_cap and n_inj >= budget_cap):
            break

        # Se WOR e popolazione esaurita, stop
        if use_wor and N == 1 and k1_order is not None and k1_ptr >= len(k1_order):
            break
        if use_wor and N > 1 and seen is not None and len(seen) >= int(N_pop):
            break

    # ------------------ FINAL STATS & SAVE ------------------
    avg_frcrit = (sum_fr / n_inj) if n_inj else 0.0
    half_norm = halfwidth_normal_fpc(avg_frcrit, n_inj, conf, (N_pop if use_fpc else None))
    w_low, w_high, w_half, w_se = wald_ci(avg_frcrit, n_inj, conf, (N_pop if use_fpc else None))
    wl_low, wl_high, wl_half = (wilson_ci_fpc(avg_frcrit, n_inj, conf, N_pop) if use_fpc
                                else wilson_ci(avg_frcrit, n_inj, conf))
    sample_std = math.sqrt(m2_frcrit / (count_fr - 1)) if count_fr > 1 else 0.0

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
    design = "WOR" if use_wor else "WR"
    with open(output_file, "w") as f:
        f.write(f"[PROP/{ep_control}] K={N}  FRcrit_avg={avg_frcrit:.8f}  conf={conf}  steps={steps}  n={n_inj}\n")
        f.write(f"half_norm={half_norm:.8f}  e_goal={e_goal:.8f}  FPC={'on' if use_fpc else 'off'}  DESIGN={design}  N_pop={N_pop}\n")
        f.write(f"EP_control={ep_control}  E_wald_final={E_wald:.8f}  E_wilson_final={E_wil:.8f}\n")
        f.write(f"Wald_CI({int(conf*100)}%):   [{w_low:.8f}, {w_high:.8f}]   half={w_half:.8f}   se={w_se:.8f}   FPC={'on' if use_fpc else 'off'}\n")
        f.write(f"Wilson_CI({int(conf*100)}%): [{wl_low:.8f}, {wl_high:.8f}]  half={wl_half:.8f}  FPC={'on' if use_fpc else 'off'}\n")
        f.write(f"SampleStdDev_FRcrit_across_injections: s={sample_std:.8f} (n={n_inj})\n")
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
        f.write("\n[E_seq] n, E_wald, E_wilson\n")
        for n_i, ew_i, ei_i in E_seq:
            f.write(f"{n_i},{ew_i:.8f},{ei_i:.8f}\n")
        f.write(f"\n[E_final] E_wald={E_seq[-1][1]:.8f}  E_wilson={E_seq[-1][2]:.8f}\n\n")

        f.write(f"Top-{min(100, len(top_sorted))} worst injections (proposed iter EP)\n")
        for rank, (frcrit, inj, faults, bias) in enumerate(top_sorted, 1):
            desc = ", ".join(f"{ft.layer_name}[{ft.tensor_index}] bit{ft.bits[0]}" for ft in faults)
            f.write(
                f"{rank:3d}) Inj {inj:6d} | FRcrit={frcrit:.6f} | "
                f"maj={bias['maj_cls']}@{bias['maj_share']:.2f} Δmax={bias['delta_max']:.3f} "
                f"KL={bias['kl']:.3f} | {desc}\n"
            )

    print(f"[PROP/{ep_control}] K={N}  avgFRcrit={avg_frcrit:.6f}  Ew_final={E_wald:.6f}  Ei_final={E_wil:.6f}  n={n_inj}  DESIGN={design} → {output_file}")
    return avg_frcrit, half_norm, n_inj, steps, top_sorted, output_file

# ============================= Dispatcher: SOLO STATISTICA (EP) =============================
def run_fault_injection(
    model,
    device,
    test_loader,
    clean_by_batch,
    baseline_hist,
    baseline_dist,
    num_classes,
    N,                                # number of simultaneous faults (K)
    MAX_FAULTS=0,                     # ignorato: niente ramo esaustivo
    save_dir="results_minimal",
    seed=None,
    exhaustive_up_to_n=-1,            # ignorato: niente ramo esaustivo
    use_fpc: bool = False,            # non usato: policy automatica
    N_pop: Optional[int] = None,      # non usato: calcolato qui
    ep_control: str = "wilson",
    prefix_tag: str = "_EP"
):
    """
    Dispatcher (solo ramo statistico EP):
      - costruisce la fault list;
      - decide automaticamente FPC in base a n_inf/N_pop (regola 5% stretta);
      - lancia la campagna iterativa EP (+FPC se attivo) e salva i risultati.
    """
    t0 = time.time()

    dataset = SETTINGS.DATASET_NAME
    net_name = SETTINGS.NETWORK_NAME
    bs = getattr(test_loader, "batch_size", None) or SETTINGS.BATCH_SIZE
    save_path = os.path.join(save_dir, dataset, net_name, f"batch_{bs}", "minimal_iterEP")
    os.makedirs(save_path, exist_ok=True)

    # Build full pool di fault elementari (layer, index, bit)
    all_faults = _build_all_faults(model, as_list=True)
    num_faults = len(all_faults)
    if num_faults == 0:
        raise RuntimeError("Empty fault list: cannot proceed.")
    if N > num_faults:
        print(f"[WARN] K={N} > available_fault_sites={num_faults}. Clamping K to {num_faults}.")
        N = num_faults

    # --- Policy automatica (regola 5% stretta) ---
    e_goal = getattr(SETTINGS, "STAT_EPS", 0.005)
    conf   = getattr(SETTINGS, "STAT_CONF", 0.95)
    p0     = getattr(SETTINGS, "STAT_P0", 0.5)
    exhaustive_cap = None  # nessun ramo esaustivo

    total_possible, use_fpc_auto, _, n_inf, n_fpc, ratio = \
        decide_sampling_policy(num_faults, N, e_goal, conf, p0, exhaustive_cap)

    design = "WOR" if use_fpc_auto else "WR"
    print(f"[POLICY] K={N} | N_pop={total_possible} | n_inf={n_inf} | ratio={ratio:.3%} | FPC_auto={use_fpc_auto} | n_fpc0={n_fpc} | DESIGN={design}")

    # Sempre ramo statistico (EP)
    comb_str = _sci_format_comb(num_faults, N)
    print(
        f"[INFO] STATISTICAL (PROPOSED): C({num_faults},{N})≈{comb_str}. "
        f"Launching iterative EP (ctrl={ep_control}){' + FPC' if use_fpc_auto else ''} [{design}]."
    )
    avg_frcrit, half_norm, n_used, steps, top_sorted, out_file = run_statistical_iterative_ep_fpc_campaign(
        model, device, test_loader, clean_by_batch, all_faults, N,
        baseline_hist, baseline_dist, num_classes,
        pilot=getattr(SETTINGS, "STAT_PILOT", 200),
        e_goal=e_goal,
        conf=conf,
        block=getattr(SETTINGS, "STAT_BLOCK", 50),
        budget_cap=getattr(SETTINGS, "STAT_BUDGET_CAP", None),
        seed=seed if seed is not None else getattr(SETTINGS, "SEED", 0),
        use_fpc=use_fpc_auto,           # ON se ratio > 5%
        N_pop=total_possible,           # dimensione popolazione finita
        save_dir=save_dir, dataset_name=dataset, net_name=net_name, bs=bs,
        ep_control=ep_control,
        prefix_tag=prefix_tag
    )
    dt_min = (time.time() - t0) / 60.0
    print(f"[PROP/{ep_control}] saved {out_file} – {dt_min:.2f} min "
          f"(avg FRcrit={avg_frcrit:.6f}, half_norm={half_norm:.6f}, n={n_used}, steps={steps})")
    return avg_frcrit, (half_norm,), n_used, top_sorted, out_file

# =================================== Main ===================================

if __name__ == "__main__":
    random.seed(getattr(SETTINGS, "SEED", 0))
    np.random.seed(getattr(SETTINGS, "SEED", 0))
    torch.manual_seed(getattr(SETTINGS, "SEED", 0))

    ds_up  = SETTINGS.DATASET.upper()
    net_lo = SETTINGS.NETWORK.lower()

    # Build + clean pass
    if ds_up == "CIFAR10" and net_lo in ("resnet20", "resnet-20"):
        print("[BOOT] Using pretrained ResNet20@CIFAR10 INT8 (fbgemm).")
        model, device, test_loader, clean_by_batch, baseline_hist, baseline_dist, num_classes = \
            build_resnet20_cifar10_pretrained_int8_and_clean()
        ONLY_WILSON = True   # SOLO Wilson su ResNet20
    else:
        model, device, test_loader, clean_by_batch, baseline_hist, baseline_dist, num_classes = \
            build_and_quantize_once()
        ONLY_WILSON = False  # Wald disponibile sugli altri

    K_LIST = getattr(SETTINGS, "K_LIST", [1, 2, 3])

    for N in K_LIST:
        if ONLY_WILSON:
            avg, half_tuple, n_used, _, out_file = run_fault_injection(
                model=model,
                device=device,
                test_loader=test_loader,
                clean_by_batch=clean_by_batch,
                baseline_hist=baseline_hist,
                baseline_dist=baseline_dist,
                num_classes=num_classes,
                N=N,
                MAX_FAULTS=getattr(SETTINGS, "MAX_FAULTS", 0),
                save_dir="results_minimal",
                seed=getattr(SETTINGS, "SEED", 0),
                exhaustive_up_to_n=getattr(SETTINGS, "EXHAUSTIVE_UP_TO_N", -1),
                ep_control="wilson",
                prefix_tag="_EPwilson"
            )
            half = _unpack_half(half_tuple)
            print(f"[DONE/WILSON] N={N} | FR={avg:.6g} | half={half:.6g} | injections={n_used} | file={out_file}")
        else:
            # Wald
            avg_w, half_tuple_w, n_w, _, out_w = run_fault_injection(
                model=model,
                device=device,
                test_loader=test_loader,
                clean_by_batch=clean_by_batch,
                baseline_hist=baseline_hist,
                baseline_dist=baseline_dist,
                num_classes=num_classes,
                N=N,
                MAX_FAULTS=getattr(SETTINGS, "MAX_FAULTS", 0),
                save_dir="results_minimal",
                seed=getattr(SETTINGS, "SEED", 0),
                exhaustive_up_to_n=getattr(SETTINGS, "EXHAUSTIVE_UP_TO_N", -1),
                ep_control="wald",
                prefix_tag="_EPwald"
            )
            half_w = _unpack_half(half_tuple_w)
            print(f"[DONE/WALD] N={N} | FR={avg_w:.6g} | half={half_w:.6g} | injections={n_w} | file={out_w}")

            # Wilson (con FPC auto)
            avg_i, half_tuple_i, n_i, _, out_i = run_fault_injection(
                model=model,
                device=device,
                test_loader=test_loader,
                clean_by_batch=clean_by_batch,
                baseline_hist=baseline_hist,
                baseline_dist=baseline_dist,
                num_classes=num_classes,
                N=N,
                MAX_FAULTS=getattr(SETTINGS, "MAX_FAULTS", 0),
                save_dir="results_minimal",
                seed=getattr(SETTINGS, "SEED", 0),
                exhaustive_up_to_n=getattr(SETTINGS, "EXHAUSTIVE_UP_TO_N", -1),
                ep_control="wilson",
                prefix_tag="_EPwilson"
            )
            half_i = _unpack_half(half_tuple_i)
            print(f"[DONE/WILSON] N={N} | FR={avg_i:.6g} | half={half_i:.6g} | injections={n_i} | file={out_i}")
