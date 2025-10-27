#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
WineMLP — Accuracy vs BER (tutti i grafici da tabella + n=1 random)

Richiede un CSV "summary" e un CSV "n1" – OPPURE può generare il CSV n=1 con --autogen-n1.

1) Tabella “summary” con (header esatti, ordine libero):
   K, n_wilson, FR_wilson, FR_wilson_low, FR_wilson_high,
   n_ep, FR_ep, wald_low, wald_high,
   N_FPC, FR_onestep, EPS_onestep
   (Opzionali: n_ep_wilson, FR_ep_wilson, WILSON_low, WILSON_high)

2) Punti n=1 random: K,rep,fr_n1 (con 3 repliche: rep ∈ {1,2,3})

Esempi:
  (solo plotting, CSV già pronti)
  python acc_ber.py \
    --table-csv plots/acc_ber/summary_table.csv \
    --n1-csv plots/acc_ber/n1_random_points.csv \
    --ci-n1 both --M 768 --z 1.96 --n19000 19000

  (genera n=1 dentro lo script + plotting)
  python acc_ber.py \
    --table-csv plots/acc_ber/summary_table.csv \
    --autogen-n1 --n1-K-list 1,2,3,4,5,6,7,8,9,10,50,100,150,384,575,768 \
    --n1-reps 3 --n1-seed 0 \
    --n1-out plots/acc_ber/n1_random_points.csv \
    --ci-n1 both --M 768
"""

import os
import math
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================ Costanti di stile (GLOBALI) ============================

# stile base SENZA marker (il marker viene passato nelle funzioni di plot)
LINE_BASE = dict(
    color="k", linewidth=1.6,
    markeredgecolor="k", markerfacecolor="white",
    markeredgewidth=1.3, zorder=3
)

COL_WIL_TABLE  = "#2ca02c"  # grafico 2 (Wilson n_wilson)
COL_EP_WALD    = "#d62728"  # grafico 3 (EP con Wald)
COL_EP_WILSON  = "#8c564b"  # grafico 3b (EP con Wilson) opzionale
COL_WALD_19000 = "#9467bd"  # grafico 4 (Wald n costante)
COL_WALD_FPC   = "#ff7f0e"  # grafico 5 (One-Step FPC)
COL_N1_WIL     = "#1f77b4"  # n=1 Wilson
COL_N1_WALD    = "#17becf"  # n=1 Wald

# ============================ Statistiche ============================

def wilson_ci(p_hat: float, n: int, z: float = 1.96):
    if n <= 0: return 0.0, 1.0
    p = float(min(max(p_hat, 1e-12), 1.0 - 1e-12))
    denom  = 1.0 + (z*z)/n
    center = (p + (z*z)/(2*n)) / denom
    half   = (z * math.sqrt((p*(1.0 - p)/n) + (z*z)/(4*n*n))) / denom
    return max(0.0, center - half), min(1.0, center + half)

def wald_ci(p_hat: float, n: int, z: float = 1.96):
    if n <= 0: return 0.0, 1.0
    p = float(min(max(p_hat, 1e-12), 1.0 - 1e-12))
    half = z * math.sqrt(p * (1.0 - p) / n)
    return max(0.0, p - half), min(1.0, p + half)

def _nonneg_yerr(center, low, high):
    center = np.asarray(center, dtype=float)
    low = np.minimum(np.asarray(low, dtype=float), center)
    high = np.maximum(np.asarray(high, dtype=float), center)
    neg = center - low
    pos = high - center
    neg[neg < 0] = 0; pos[pos < 0] = 0
    return np.vstack([neg, pos])

# -------------------- Stile assi --------------------
def _style_axes():
    plt.xlabel("BER = K / M")
    plt.ylabel("Accuracy (1 − FR) [%]")
    plt.ylim(0, 100)
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    plt.gca().set_axisbelow(True)

# -------------------- Helper grafici --------------------
def _acc_from_fr(fr): 
    return (1.0 - np.asarray(fr, dtype=float)) * 100.0

def _plot_series_with_bands(x, y_acc, low_acc, high_acc, color, marker, label, title, outpath):
    yerr = _nonneg_yerr(y_acc, low_acc, high_acc)
    fig = plt.figure(figsize=(7, 5), dpi=140)
    # passa il marker QUI (LINE_BASE non lo ha)
    plt.plot(x, y_acc, marker=marker, label=label, **LINE_BASE)
    plt.errorbar(x, y_acc, yerr=yerr, fmt="none",
                 ecolor=color, elinewidth=2.0, capsize=5, capthick=2.0, alpha=0.95, zorder=2)
    _style_axes(); plt.title(title); plt.legend()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.tight_layout(); fig.savefig(outpath); plt.close(fig)
    return outpath

# ============================ Plot: dalla tabella ============================

def plot_from_table(df_tab: pd.DataFrame, M: int, outdir: str, z: float, n19000: int, title_prefix: str):
    saved = []

    # Ordina e calcola BER
    df = df_tab.copy()
    for c in ["K","n_wilson","FR_wilson","FR_wilson_low","FR_wilson_high",
              "n_ep","FR_ep","wald_low","wald_high",
              "N_FPC","FR_onestep","EPS_onestep",
              "n_ep_wilson","FR_ep_wilson","WILSON_low","WILSON_high"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("K").reset_index(drop=True)
    x = (df["K"].astype(float) / float(M)).values

    # ===== Grafico 2: Wilson (n = n_wilson) =====
    y_acc  = _acc_from_fr(df["FR_wilson"].values)
    low_a  = _acc_from_fr(df["FR_wilson_high"].values)  # high FR -> low Acc
    high_a = _acc_from_fr(df["FR_wilson_low"].values)
    p2 = os.path.join(outdir, f"{title_prefix}_accuracy_vs_BER_Wilson_nTable.png")
    saved.append(_plot_series_with_bands(
        x, y_acc, low_a, high_a, COL_WIL_TABLE, "o",
        label="Wilson (n = n_wilson)", 
        title=f"{title_prefix} — Accuracy vs BER (Wilson 95% CI, n = n_injections)",
        outpath=p2
    ))

    # ===== Grafico 3: EP (punti = FR_ep; barre = Wald da tabella) =====
    if {"FR_ep","wald_low","wald_high"}.issubset(df.columns):
        y_acc  = _acc_from_fr(df["FR_ep"].values)
        low_a  = _acc_from_fr(df["wald_high"].values)
        high_a = _acc_from_fr(df["wald_low"].values)
        p3 = os.path.join(outdir, f"{title_prefix}_accuracy_vs_BER_EP_Wald.png")
        saved.append(_plot_series_with_bands(
            x, y_acc, low_a, high_a, COL_EP_WALD, "s",
            label="EP (Wald 95% CI)",
            title=f"{title_prefix} — Accuracy vs BER (EP, Wald 95% CI)",
            outpath=p3
        ))

    # (Opzionale) EP-Wilson se presente in tabella
    if {"FR_ep_wilson","WILSON_low","WILSON_high"}.issubset(df.columns):
        y_acc  = _acc_from_fr(df["FR_ep_wilson"].values)
        low_a  = _acc_from_fr(df["WILSON_high"].values)
        high_a = _acc_from_fr(df["WILSON_low"].values)
        p3b = os.path.join(outdir, f"{title_prefix}_accuracy_vs_BER_EP_Wilson.png")
        saved.append(_plot_series_with_bands(
            x, y_acc, low_a, high_a, COL_EP_WILSON, "D",
            label="EP (Wilson 95% CI)",
            title=f"{title_prefix} — Accuracy vs BER (EP, Wilson 95% CI)",
            outpath=p3b
        ))

    # ===== Grafico 4: Wald n = n19000 costante (sui FR_wilson) =====
    acc  = _acc_from_fr(df["FR_wilson"].values)
    low  = []; high = []
    for p in df["FR_wilson"].values:
        lo, hi = wald_ci(float(p), n=n19000, z=z)
        low.append((1.0 - hi) * 100.0)
        high.append((1.0 - lo) * 100.0)
    p4 = os.path.join(outdir, f"{title_prefix}_accuracy_vs_BER_Wald_n{n19000}.png")
    saved.append(_plot_series_with_bands(
        x, acc, np.array(low), np.array(high), COL_WALD_19000, "d",
        label=f"Wald (n = {n19000})",
        title=f"{title_prefix} — Accuracy vs BER (Wald 95% CI, n = {n19000})",
        outpath=p4
    ))

    # ===== Grafico 5: One-Step FPC (punti FR_onestep; barre ±EPS) =====
    if {"FR_onestep","EPS_onestep"}.issubset(df.columns):
        fr  = df["FR_onestep"].values.astype(float)
        eps = df["EPS_onestep"].values.astype(float)
        acc = (1.0 - fr) * 100.0
        low = (1.0 - np.clip(fr + eps, 0.0, 1.0)) * 100.0
        high= (1.0 - np.clip(fr - eps, 0.0, 1.0)) * 100.0
        p5 = os.path.join(outdir, f"{title_prefix}_accuracy_vs_BER_OneStepFPC.png")
        saved.append(_plot_series_with_bands(
            x, acc, low, high, COL_WALD_FPC, "^",
            label="One-Step (FPC, ε tabella)",
            title=f"{title_prefix} — Accuracy vs BER (One-Step FPC, 95%)",
            outpath=p5
        ))

    return saved

# ============================ n=1: generazione opzionale ============================

def autogen_n1_points(Ks, reps=3, seed=0, out_csv=None):
    """
    Genera punti n=1 (una iniezione con K fault elementari scelti a caso) usando il tuo stack.
    Ritorna un DataFrame (K, rep, fr_n1). Se out_csv è dato, salva anche il CSV.
    Richiede i moduli: faultManager.*, utils.*, SETTINGS.
    """
    import random
    from itertools import product
    import torch
    from tqdm import tqdm

    from faultManager.WeightFault import WeightFault
    from faultManager.WeightFaultInjector import WeightFaultInjector
    from utils import get_loader, load_from_dict, get_network, load_quantized_model, save_quantized_model
    import SETTINGS

    def _get_quant_weight(module: torch.nn.Module) -> torch.Tensor:
        if hasattr(module, "weight"):
            try: return module.weight()
            except Exception: pass
        if hasattr(module, "_packed_params") and hasattr(module._packed_params, "_weight_bias"):
            w, _ = module._packed_params._weight_bias()
            return w
        raise RuntimeError("Impossibile ottenere il peso quantizzato dal modulo.")

    def _build_all_faults(model):
        faults = []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.quantized.Linear, torch.nn.quantized.Conv2d)):
                try: w = _get_quant_weight(module)
                except Exception: continue
                for idx in product(*[range(s) for s in w.shape]):
                    for bit in range(8):
                        faults.append((name, idx, bit))
        return faults

    def _build_and_quantize_once():
        torch.backends.quantized.engine = "fbgemm"
        device = torch.device("cpu")
        ds = getattr(SETTINGS, "DATASET_NAME", getattr(SETTINGS, "DATASET", ""))
        net = getattr(SETTINGS, "NETWORK_NAME", getattr(SETTINGS, "NETWORK", ""))

        train_loader, _, test_loader = get_loader(
            dataset_name=ds,
            batch_size=getattr(SETTINGS, "BATCH_SIZE", 64),
            network_name=net
        )
        qmodel, qpath = load_quantized_model(ds, net, device="cpu", engine="fbgemm")
        if qmodel is not None:
            model = qmodel
            print(f"[PTQ] caricato quantizzato: {qpath}")
        else:
            model = get_network(net, device, ds).to(device).eval()
            ckpt = f"./trained_models/{ds}_{net}_trained.pth"
            if os.path.exists(ckpt):
                load_from_dict(model, device, ckpt)
                print("[CKPT] checkpoint float caricato")
            if hasattr(model, "quantize_model"):
                model.quantize_model(calib_loader=train_loader)
                model.eval()
                save_quantized_model(model, ds, net, engine="fbgemm")
                print("[PTQ] quantizzato e salvato")
        clean_by_batch = []
        with torch.inference_mode():
            for xb, _ in test_loader:
                logits = model(xb.to(device))
                clean_by_batch.append(torch.argmax(logits, dim=1).cpu())
        total_samples = sum(len(t) for t in clean_by_batch)
        if total_samples == 0:
            raise RuntimeError("Test loader vuoto.")
        return model, device, test_loader, clean_by_batch, total_samples

    def _eval_frcrit_for_combo(model, device, test_loader, clean_by_batch, combo, inj_id, total_samples):
        injector = WeightFaultInjector(model)
        faults = [WeightFault(injection=inj_id, layer_name=ln, tensor_index=ti, bits=[bt])
                  for (ln, ti, bt) in combo]
        mismatches = 0
        try:
            injector.inject_faults(faults, 'bit-flip')
            with torch.inference_mode():
                for batch_i, (xb, _) in enumerate(test_loader):
                    pred_f = torch.argmax(model(xb.to(device)), dim=1).cpu().numpy()
                    clean_pred = clean_by_batch[batch_i].numpy()
                    mismatches += int((pred_f != clean_pred).sum())
        finally:
            injector.restore_golden()
        return mismatches / float(total_samples)

    # ---- esecuzione ----
    Ks = list(sorted(set(int(k) for k in Ks)))
    model, device, test_loader, clean_by_batch, total_samples = _build_and_quantize_once()
    all_faults = _build_all_faults(model)
    M = len(all_faults)
    print(f"[n=1] Elementary faults M = {M}")

    rows = []
    inj_id = 0
    for rep in range(1, reps + 1):
        rnd = random.Random(seed + 10007 * rep)
        for K in Ks:
            if K > M:
                print(f"[WARN] K={K} > M={M}: salto.")
                continue
            idxs = rnd.sample(range(M), K)
            combo = [all_faults[i] for i in idxs]
            inj_id += 1
            fr = _eval_frcrit_for_combo(model, device, test_loader, clean_by_batch,
                                        combo, inj_id, total_samples)
            rows.append((K, rep, float(fr)))
            print(f"[n=1] rep={rep:2d} K={K:4d} → FR={fr:.6f}")

    df = pd.DataFrame(rows, columns=["K","rep","fr_n1"]).sort_values(["rep","K"]).reset_index(drop=True)
    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"[n=1] salvato: {out_csv}")
    return df

# ============================ Plot: n=1 random (3 repliche) ============================

def plot_n1_random(df_n1: pd.DataFrame, M: int, outdir: str, z: float, title_prefix: str, ci_kind: str):
    saved = []
    df = df_n1.copy()
    for c in ["K","rep","fr_n1"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    reps = sorted(df["rep"].dropna().unique().tolist())

    for rep in reps:
        sub = df[df["rep"] == rep].sort_values("K")
        x = (sub["K"].astype(float) / float(M)).values
        p = sub["fr_n1"].astype(float).values

        def _do(ci_label, color):
            lo = []; hi = []
            for pi in p:
                if ci_label == "wilson": l, h = wilson_ci(pi, n=1, z=z)
                else:                     l, h = wald_ci  (pi, n=1, z=z)
                lo.append(l); hi.append(h)
            acc_pts  = (1.0 - p)  * 100.0
            acc_low  = (1.0 - np.array(hi)) * 100.0
            acc_high = (1.0 - np.array(lo)) * 100.0
            yerr = _nonneg_yerr(acc_pts, acc_low, acc_high)

            fig = plt.figure(figsize=(7, 5), dpi=140)
            # linea nera che passa ESATTAMENTE per i punti della replica
            plt.plot(x, acc_pts, marker="o", label=f"rep {rep} (n=1)", **LINE_BASE)
            plt.errorbar(x, acc_pts, yerr=yerr, fmt="none",
                         ecolor=color, elinewidth=2.0, capsize=5, capthick=2.0, alpha=0.95, zorder=2)
            _style_axes()
            name = "Wilson" if ci_label=="wilson" else "Wald"
            plt.title(f"{title_prefix} — Accuracy vs BER ({name} 95% CI, n=1, replica {int(rep)})")
            plt.legend()
            fn = f"{title_prefix}_accuracy_vs_BER_{name}_n1_rep{int(rep)}.png"
            path = os.path.join(outdir, fn)
            fig.tight_layout(); fig.savefig(path); plt.close(fig)
            return path

        if ci_kind in ("wilson","both"):
            saved.append(_do("wilson", COL_N1_WIL))
        if ci_kind in ("wald","both"):
            saved.append(_do("wald",   COL_N1_WALD))
    return saved


# ============================ Main ============================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table-csv", type=str, required=True,
                    help="CSV della tabella summary (vedi header richiesti nel docstring)")
    ap.add_argument("--n1-csv", type=str, default=None,
                    help="CSV con punti n=1 random (K,rep,fr_n1). Ignorato se --autogen-n1.")
    ap.add_argument("--autogen-n1", action="store_true",
                    help="Se presente, genera i punti n=1 direttamente in questo script.")
    ap.add_argument("--n1-K-list", type=str,
                    default="1,2,3,4,5,6,7,8,9,10,50,100,150,384,575,768")
    ap.add_argument("--n1-reps", type=int, default=3)
    ap.add_argument("--n1-seed", type=int, default=0)
    ap.add_argument("--n1-out", type=str, default="plots/acc_ber/n1_random_points.csv",
                    help="Dove salvare (o leggere) i punti n=1")
    ap.add_argument("--outdir", type=str, default="plots/acc_ber")
    ap.add_argument("--M", type=int, default=768)
    ap.add_argument("--z", type=float, default=1.96)
    ap.add_argument("--n19000", type=int, default=19000)
    ap.add_argument("--ci-n1", choices=["wilson","wald","both"], default="wilson",
                    help="Tipo di CI per i grafici n=1")
    ap.add_argument("--title", type=str, default="WineMLP")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---- Carica tabella summary ----
    df_tab = pd.read_csv(args.table_csv)

    # ---- Carica o genera n=1 ----
    if args.autogen_n1:
        Ks = [int(x.strip()) for x in args.n1_K_list.split(",") if x.strip()]
        df_n1 = autogen_n1_points(Ks, reps=args.n1_reps, seed=args.n1_seed, out_csv=args.n1_out)
    else:
        if not args.n1_csv:
            # fallback: prova a leggere da --n1-out se non dato --n1-csv
            args.n1_csv = args.n1_out
        df_n1  = pd.read_csv(args.n1_csv)

    # ---- Plot ----
    saved = []
    saved += plot_from_table(df_tab, M=args.M, outdir=args.outdir, z=args.z,
                             n19000=args.n19000, title_prefix=args.title)
    saved += plot_n1_random(df_n1, M=args.M, outdir=args.outdir, z=args.z,
                            title_prefix=args.title, ci_kind=args.ci_n1)

    print("[OK] Salvato:")
    for p in saved:
        print(" -", p)

if __name__ == "__main__":
    main()
