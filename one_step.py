#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
One Step EP (non-iterativo) – calcolo *automatico* dei campioni, **con FPC**.
- Calcola M dai siti di fault del modello quantizzato.
- Per ogni K:
   N = C(M,K), ratio = (N-1)/N,
   n_inf = ceil(z^2 p0(1-p0) / e^2),
   n_fpc = ceil( N * z^2 p0(1-p0) / ( (N-1)*e^2 + z^2 p0(1-p0) ) ), clampato a [1, N],
   n_to_draw = n_fpc.
- Esegue esattamente n_to_draw iniezioni (combinazioni uniche, senza duplicati) e stampa:
   K, N, (N-1)/N, f_exhaustive (opzionale), n_inf, n_fpc, f_rate (EP), errore (EP, con FPC), n_used.

NOTE:
- FPC (“finite population correction”) rende il one-step **comparabile** con la tua tabella,
  in cui n_fpc << n_inf quando N è piccolo (es. K=1).
- L’errore riportato è l’half-width normale **con FPC** calcolato su p_hat finale.
"""

import os
import math
import random
import argparse
from itertools import product, combinations, islice
from typing import Optional, Iterable, Tuple, List

import numpy as np
import torch
from tqdm import tqdm

from faultManager.WeightFault import WeightFault
from faultManager.WeightFaultInjector import WeightFaultInjector
from utils import get_loader, load_from_dict, get_network
import SETTINGS


# ---------- stampa C(n,k) in notazione scientifica ----------
def sci_comb(n: int, k: int) -> str:
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


# ---------- helper per ottenere i pesi quantizzati ----------
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


def build_fault_sites(model, as_list=True):
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


# ---------- EP planning (CON FPC) ----------
def plan_n_one_step(z: float, e: float, p0: float, N_pop: int) -> Tuple[int, int]:
    """
    Ritorna (n_inf, n_fpc):
      n_inf  = ceil(z^2 p0(1-p0) / e^2)
      n_fpc  = ceil( N * z^2 p0(1-p0) / ( (N-1)*e^2 + z^2 p0(1-p0) ) )
    """
    A = (z*z*p0*(1.0-p0))  # = 0.9604 con z=1.96, p0=0.5
    n_inf = max(1, math.ceil(A / (e*e)))
    denom = (e*e)*(N_pop - 1.0) + A
    n_fpc = max(1, min(N_pop, int(math.ceil((N_pop * A) / denom))))
    return n_inf, n_fpc


def halfwidth_normal_fpc(p_hat: float, n: int, z: float, N_pop: int) -> float:
    """
    Half-width normale con FPC = z * sqrt( p(1-p)/n ) * sqrt((N-n)/(N-1))
    """
    p = min(max(p_hat, 1e-12), 1.0 - 1e-12)
    base = z * math.sqrt(p * (1.0 - p) / max(1, n))
    if N_pop and N_pop > 1 and 1 <= n <= N_pop:
        fpc = math.sqrt((N_pop - n) / (N_pop - 1.0))
    else:
        fpc = 1.0
    return base * fpc


# ---------- sampler di combinazioni uniche ----------
def sample_unique_combos(pool: List[tuple], K: int, n_needed: int, seed: int) -> Iterable[Tuple[tuple, ...]]:
    """
    Estrae n_needed combinazioni uniche di K fault senza duplicati cross-sample.
    - Per K=1 usa un campione senza rimpiazzo perfetto.
    - Se la frazione richiesta è alta e la popolazione è gestibile, enumera con combinations+islice.
    - Altrimenti, prova/rigetta su indici con set() e poi mappa a pool[*].
    """
    rnd = random.Random(seed)
    m = len(pool)
    assert 1 <= K <= m
    max_unique = math.comb(m, K)
    n_needed = min(n_needed, max_unique)

    if K == 1:
        for i in rnd.sample(range(m), n_needed):
            yield (pool[i],)
        return

    frac = n_needed / max_unique if max_unique > 0 else 0.0
    if frac > 0.25 and max_unique <= 1_000_000:
        for idxs in islice(combinations(range(m), K), n_needed):
            yield tuple(pool[i] for i in idxs)
        return

    seen = set()
    while len(seen) < n_needed:
        idxs = tuple(sorted(rnd.sample(range(m), K)))
        if idxs not in seen:
            seen.add(idxs)
            yield tuple(pool[i] for i in idxs)


# ---------- build + quantize + clean ----------
def build_and_quantize_once():
    torch.backends.quantized.engine = "fbgemm"
    device = torch.device("cpu")

    print(f"Loading network {SETTINGS.NETWORK_NAME} for {SETTINGS.DATASET_NAME} ...")
    model = get_network(SETTINGS.NETWORK_NAME, device, SETTINGS.DATASET_NAME)
    model.to(device).eval()
    ckpt = f"./trained_models/{SETTINGS.DATASET_NAME}_{SETTINGS.NETWORK_NAME}_trained.pth"
    if os.path.exists(ckpt):
        load_from_dict(model, device, ckpt)
        print("modello caricato")
    else:
        print(f"checkpoint non trovato: {ckpt} (proseguo senza)")

    print(f"Loading {SETTINGS.DATASET_NAME} dataset...")
    train_loader, _, test_loader = get_loader(
        dataset_name=SETTINGS.DATASET_NAME,
        batch_size=SETTINGS.BATCH_SIZE,
        network_name=SETTINGS.NETWORK_NAME
    )

    if hasattr(model, "quantize_model"):
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

    return model, device, test_loader, clean_by_batch, baseline_hist, baseline_dist, num_classes


# ---------- valutazione di una combinazione ----------
def evaluate_combo(model, device, test_loader, clean_by_batch, injector,
                   combo, inj_id, total_samples):
    faults = [WeightFault(injection=inj_id, layer_name=ln, tensor_index=ti, bits=[bt])
              for (ln, ti, bt) in combo]
    mismatches = 0
    try:
        injector.inject_faults(faults, 'bit-flip')
        with torch.inference_mode():
            for (batch_i, (xb, _)) in enumerate(test_loader):
                xb = xb.to(device)
                logits_f = model(xb)
                pred_f = torch.argmax(logits_f, dim=1).cpu().numpy()
                clean_pred = clean_by_batch[batch_i].numpy()
                mismatches += int((pred_f != clean_pred).sum())
    finally:
        injector.restore_golden()
    return mismatches / float(total_samples)


# ---------- runner one-step EP (con FPC) ----------
def run_one_step_ep_for_K(model, device, test_loader, clean_by_batch,
                          all_faults, K, z, e, p0, seed, show_progress=True):
    M = len(all_faults)
    if K < 1 or K > M:
        return None

    N_pop = math.comb(M, K)
    n_inf, n_fpc = plan_n_one_step(z=z, e=e, p0=p0, N_pop=N_pop)
    n_to_draw = n_fpc  # per la tabella devi usare FPC

    injector = WeightFaultInjector(model)
    total_samples = sum(len(t) for t in clean_by_batch)
    sum_fr = 0.0

    it = sample_unique_combos(all_faults, K, n_to_draw, seed=seed)
    it = tqdm(it, total=n_to_draw, desc=f"[OneStep] K={K}", disable=not show_progress)
    for inj_id, combo in enumerate(it, 1):
        sum_fr += evaluate_combo(model, device, test_loader, clean_by_batch, injector,
                                 combo, inj_id, total_samples)

    p_hat = sum_fr / n_to_draw if n_to_draw > 0 else 0.0
    err = halfwidth_normal_fpc(p_hat, n_to_draw, z, N_pop)
    ratio = 1.0 - (1.0 / float(N_pop)) if N_pop > 0 else 1.0

    return dict(
        K=K,
        N_pop=N_pop,
        N_pop_str=sci_comb(M, K),
        ratio=ratio,
        n_inf=n_inf,
        n_fpc=n_fpc,
        f_rate_ep=p_hat,
        err_ep=err,
        n_used=n_to_draw
    )


# ---------- (opzionale) esaustiva se N piccolo ----------
def maybe_exhaustive_f(model, device, test_loader, clean_by_batch,
                       all_faults, K, exhaustive_cap: int) -> Optional[float]:
    M = len(all_faults)
    N_pop = math.comb(M, K)
    if exhaustive_cap <= 0 or N_pop > exhaustive_cap:
        return None
    injector = WeightFaultInjector(model)
    total_samples = sum(len(t) for t in clean_by_batch)
    sum_fr = 0.0
    for inj_id, combo in enumerate(tqdm(combinations(all_faults, K), total=N_pop, desc=f"[EXA] K={K}"), 1):
        sum_fr += evaluate_combo(model, device, test_loader, clean_by_batch, injector,
                                 combo, inj_id, total_samples)
    return sum_fr / N_pop if N_pop > 0 else 0.0


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="One Step EP – CON FPC")
    ap.add_argument("--Ks", type=str, default="1,2,3,4,5,6,7,8,9,10,50,100,150,384,575,766",
                    help="Lista di K separati da virgola")
    ap.add_argument("--e", type=float, default=0.005, help="Errore target (half-width), es. 0.005")
    ap.add_argument("--z", type=float, default=1.96, help="Quantile z per conf ~95%% (1.96)")
    ap.add_argument("--p0", type=float, default=0.5, help="p0 conservativo per pianificazione (0.5)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exhaustive-cap", type=int, default=0, help="Se >0 e N<=cap, calcola anche f_exhaustive")
    args = ap.parse_args()

    # build + clean
    model, device, test_loader, clean_by_batch, _, _, _ = build_and_quantize_once()

    # siti di fault (M)
    all_faults = build_fault_sites(model, as_list=True)
    M = len(all_faults)
    print(f"[INFO] Total Faults (M) = {M}")

    # parse Ks
    Ks = [int(k.strip()) for k in args.Ks.split(",") if k.strip()]

    # header
    print("K\tN\t(N-1)/N\tf_exhaustive\t n\t n (FPC)\t f-rate (EP)\t errore eps\t n_used")

    for K in Ks:
        if K < 1 or K > M:
            print(f"{K}\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA")
            continue

        out = run_one_step_ep_for_K(
            model, device, test_loader, clean_by_batch,
            all_faults, K,
            z=args.z, e=args.e, p0=args.p0,
            seed=args.seed, show_progress=True
        )
        if out is None:
            print(f"{K}\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA")
            continue

        f_ex = maybe_exhaustive_f(
            model, device, test_loader, clean_by_batch,
            all_faults, K, exhaustive_cap=args.exhaustive_cap
        )
        f_ex_str = f"{f_ex:.8f}" if f_ex is not None else "NA"

        print(f"{out['K']}\t{out['N_pop_str']}\t{out['ratio']:.9f}\t{f_ex_str}\t"
              f"{out['n_inf']}\t{out['n_fpc']}\t{out['f_rate_ep']:.8f}\t{out['err_ep']:.8f}\t{out['n_used']}")

if __name__ == "__main__":
    main()
