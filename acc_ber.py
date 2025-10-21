#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Accuracy vs BER (WineMLP) con barre di errore.
- Grafico 1: Wilson 95% con n = 1 (didattico)
- Grafico 1b: Wald 95% con n = 1 (punti = FR_ep)
- Grafico 2: Wilson 95% con n = n_wilson (dalla tabella)
- Grafico 3: EP/Wald 95% (punti = f-rate EP, barre = intervallo di Wald da tabella)
- Grafico 4: Wald 95% con n = 19000 costante (punti = FR_wilson)
- Grafico 5: One-Step FPC (punti = FR_one_step, barre = ±epsilon da tabella)

Nessuna campagna: usa solo i dati forniti.
Salva in plots/acc_ber/.
"""

import os
import math
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# ---------- Parametri ----------
M_DEFAULT = 768  # Total Faults (bit elementari) per WineMLP

# Tabella embedded:
# (K, FR_exhaustive, n_wilson, FR_wilson, FR_wilson_low, FR_wilson_high,
#  n_ep, FR_ep, wald_low, wald_high)
ROWS = [
    (1,   0.00663098, 1300, 0.00739316, 0.00397337, 0.01371576, 1150,  0.00737520, 0.00242997, 0.01232043),
    (2,   0.01280599, 2150, 0.01335056, 0.00929436, 0.01914274, 2000,  0.01318519, 0.00818597, 0.01818440),
    (3,   0.01305633, 2850, 0.01829760, 0.01398614, 0.02390591, 2800,  0.01842593, 0.01344450, 0.02340735),
    (4,   float("nan"), 3700, 0.02438939, 0.01989041, 0.02987497, 3650,  0.02422628, 0.01923827, 0.02921429),
    (5,   float("nan"), 4450, 0.02949230, 0.02491289, 0.03488337, 4400,  0.02941498, 0.02442234, 0.03440763),
    (6,   float("nan"), 5150, 0.03419274, 0.02956643, 0.03951346, 5085,  0.03424014, 0.02924195, 0.03923833),
    (7,   float("nan"), 5800, 0.03914751, 0.03445349, 0.04445162, 5787,  0.03913305, 0.03413693, 0.04412917),
    (8,   float("nan"), 6500, 0.04389459, 0.03917785, 0.04915013, 6450,  0.04382142, 0.03882580, 0.04881703),
    (9,   float("nan"), 7050, 0.04791437, 0.04317009, 0.05315107, 7050,  0.04790386, 0.04291861, 0.05288911),
    (10,  float("nan"), 7650, 0.05231905, 0.04755011, 0.05753739, 7627,  0.05233843, 0.04734020, 0.05733666),
    (50,  float("nan"),21400, 0.16675320, 0.16181882, 0.17180721, 21357, 0.16678372, 0.16178405, 0.17178340),
    (100, float("nan"),30750, 0.27625053, 0.27128091, 0.28127605, 30727, 0.27627892, 0.27127908, 0.28127875),
    (150, float("nan"),36300, 0.38167330, 0.37668851, 0.38668313, 36288, 0.38165560, 0.37665727, 0.38665394),
    (384, float("nan"),34350, 0.66263410, 0.65761605, 0.66761579, 34370, 0.66261328, 0.65761454, 0.66761202),
    (575, float("nan"),38050, 0.55170585, 0.54670382, 0.55669743, 38108, 0.54497646, 0.53997665, 0.54997627),
    (766, float("nan"),28400, 0.24469288, 0.23972764, 0.24972718, 28403, 0.24471985, 0.23971993, 0.24971976),
]

# n (with Pop. Correcting Factor) allineati a ROWS (informativo)
N_FPC_LIST = [
    753, 33984, 38396, 38416, 38416, 38416, 38416, 38416,
    38416, 38416, 38416, 38416, 38416, 38416, 38416, 33984,
]

# One-Step (FPC) – f-rate ed epsilon dalla tabella, allineati a ROWS
ONESTEP_FR_LIST = [
    0.00671389, 0.01284973, 0.01865163, 0.02421011, 0.02944327, 0.03438768,
    0.03909829, 0.04377275, 0.04818112, 0.05236630, 0.16679055, 0.27665460,
    0.38221054, 0.66288352, 0.54930623, 0.24475516,
]
ONESTEP_EPS_LIST = [
    0.00081570, 0.00112625, 0.00135290, 0.00153701, 0.00169045, 0.00182223,
    0.00193829, 0.00204589, 0.00214149, 0.00222765, 0.00372789, 0.00447344,
    0.00429879, 0.00472725, 0.00497563, 0.00429939,
]

# ---------- Statistiche ----------
def wilson_ci(p_hat: float, n: int, z: float = 1.96):
    if n <= 0:
        return 0.0, 1.0
    p = min(max(float(p_hat), 1e-12), 1.0 - 1e-12)
    denom  = 1.0 + (z*z)/n
    center = (p + (z*z)/(2*n)) / denom
    half   = (z * math.sqrt((p*(1.0 - p)/n) + (z*z)/(4*n*n))) / denom
    return max(0.0, center - half), min(1.0, center + half)

def wald_ci(p_hat: float, n: int, z: float = 1.96):
    if n <= 0:
        return 0.0, 1.0
    p = min(max(float(p_hat), 1e-12), 1.0 - 1e-12)
    half = z * math.sqrt(p * (1.0 - p) / n)
    return max(0.0, p - half), min(1.0, p + half)

def _nonneg_yerr(center, low, high):
    low = np.minimum(low, center)
    high = np.maximum(high, center)
    neg = center - low
    pos = high - center
    neg[neg < 0] = 0
    pos[pos < 0] = 0
    return np.vstack([neg, pos])

# ---------- Stile & helper ----------
def _style_axes():
    plt.xlabel("BER = K / M")
    plt.ylabel("Accuracy (1 − FR) [%]")
    plt.ylim(0, 100)
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    plt.gca().set_axisbelow(True)

# linea/marker base (sotto)
LINE_BASE = dict(color="k", linewidth=1.2, markeredgecolor="k",
                 markerfacecolor="white", markeredgewidth=1.2, zorder=2)

# helper: prima la curva, poi gli error bar “in rilievo”
def plot_with_errors(x, y, yerr, marker, err_color, label):
    line_kw = LINE_BASE.copy()
    line_kw["marker"] = marker
    plt.plot(x, y, **line_kw, label=label)
    plt.errorbar(x, y, yerr=yerr, fmt="none",
                 ecolor=err_color, elinewidth=2.2, capsize=6, capthick=2.2,
                 alpha=0.98, zorder=3)

# Colori dei soli error bar
COL_WIL_N1     = "#1f77b4"  # blu
COL_WALD_N1_EP = "#17becf"  # ciano
COL_WIL_TABLE  = "#2ca02c"  # verde
COL_EP_WALD    = "#d62728"  # rosso
COL_WALD_19000 = "#9467bd"  # viola
COL_WALD_FPC   = "#ff7f0e"  # arancio

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=M_DEFAULT)
    ap.add_argument("--outdir", type=str, default="plots/acc_ber")
    ap.add_argument("--svg", action="store_true")
    ap.add_argument("--z", type=float, default=1.96)
    ap.add_argument("--n19000", type=int, default=19000)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = pd.DataFrame(
        ROWS,
        columns=["K","FR_exhaustive","n_wilson","FR_wilson","FR_wilson_low","FR_wilson_high",
                 "n_ep","FR_ep","wald_low","wald_high"],
    ).sort_values("K").reset_index(drop=True)

    # BER
    df["BER"] = df["K"] / float(args.M)

    # Punti in Accuracy (da Wilson tabellato)
    df["Acc_point_wilson"]    = (1.0 - df["FR_wilson"]) * 100.0
    df["Acc_low_tab_wilson"]  = (1.0 - df["FR_wilson_high"]) * 100.0
    df["Acc_high_tab_wilson"] = (1.0 - df["FR_wilson_low"])  * 100.0

    # ===== Grafico 1: Wilson n=1 (didattico) =====
    x = df["BER"].values
    acc_points_1, acc_low_1, acc_high_1 = [], [], []
    for p in df["FR_wilson"].values:
        lo, hi = wilson_ci(float(p), n=1, z=args.z)
        acc_points_1.append((1.0 - p) * 100.0)
        acc_low_1.append((1.0 - hi) * 100.0)
        acc_high_1.append((1.0 - lo) * 100.0)
    acc_points_1 = np.array(acc_points_1)
    yerr_1 = _nonneg_yerr(acc_points_1, np.array(acc_low_1), np.array(acc_high_1))

    fig1 = plt.figure(figsize=(7, 5), dpi=140)
    plot_with_errors(x, acc_points_1, yerr_1, marker="o", err_color=COL_WIL_N1, label="Wilson (n=1)")
    _style_axes(); plt.title("WineMLP — Accuracy vs BER (Wilson 95% CI, n = 1)"); plt.legend()
    p1 = os.path.join(args.outdir, "WineMLP_accuracy_vs_BER_Wilson_n1.png")
    fig1.tight_layout(); fig1.savefig(p1); 
    if args.svg: fig1.savefig(p1.replace(".png",".svg"))
    plt.close(fig1)

    # ===== Grafico 1b: Wald n=1 (punti = FR_ep) =====
    if not df["FR_ep"].isna().all():
        acc_points_wald1, acc_low_wald1, acc_high_wald1 = [], [], []
        for p in df["FR_ep"].values:
            lo, hi = wald_ci(float(p), n=1, z=args.z)
            acc_points_wald1.append((1.0 - p) * 100.0)
            acc_low_wald1.append((1.0 - hi) * 100.0)
            acc_high_wald1.append((1.0 - lo) * 100.0)
        acc_points_wald1 = np.array(acc_points_wald1)
        yerr_wald1 = _nonneg_yerr(acc_points_wald1, np.array(acc_low_wald1), np.array(acc_high_wald1))

        fig1b = plt.figure(figsize=(7, 5), dpi=140)
        plot_with_errors(x, acc_points_wald1, yerr_wald1, marker="v",
                         err_color=COL_WALD_N1_EP, label="Wald (n=1, p̂ = EP)")
        _style_axes(); plt.title("WineMLP — Accuracy vs BER (Wald 95% CI, n = 1, p̂ = EP)"); plt.legend()
        p1b = os.path.join(args.outdir, "WineMLP_accuracy_vs_BER_Wald_n1_EP.png")
        fig1b.tight_layout(); fig1b.savefig(p1b);
        if args.svg: fig1b.savefig(p1b.replace(".png",".svg"))
        plt.close(fig1b)
    else:
        p1b = None

    # ===== Grafico 2: Wilson n = n_wilson (tabella) =====
    acc_points_n = df["Acc_point_wilson"].values
    yerr_n = _nonneg_yerr(acc_points_n,
                          df["Acc_low_tab_wilson"].values,
                          df["Acc_high_tab_wilson"].values)

    fig2 = plt.figure(figsize=(7, 5), dpi=140)
    plot_with_errors(x, acc_points_n, yerr_n, marker="o", err_color=COL_WIL_TABLE, label="Wilson (n = n_wilson)")
    _style_axes(); plt.title("WineMLP — Accuracy vs BER (Wilson 95% CI, n = n_injections)"); plt.legend()
    p2 = os.path.join(args.outdir, "WineMLP_accuracy_vs_BER_Wilson_nTable.png")
    fig2.tight_layout(); fig2.savefig(p2);
    if args.svg: fig2.savefig(p2.replace(".png",".svg"))
    plt.close(fig2)

    # ===== Grafico 3: EP (punti = FR_ep; barre = Wald da tabella) =====
    if not df["FR_ep"].isna().all():
        acc_points_ep = (1.0 - df["FR_ep"].values) * 100.0
        acc_low_ep    = (1.0 - df["wald_high"].values) * 100.0
        acc_high_ep   = (1.0 - df["wald_low"].values)  * 100.0
        yerr_ep = _nonneg_yerr(acc_points_ep, acc_low_ep, acc_high_ep)

        fig3 = plt.figure(figsize=(7, 5), dpi=140)
        plot_with_errors(x, acc_points_ep, yerr_ep, marker="s", err_color=COL_EP_WALD, label="EP (Wald 95% CI)")
        _style_axes(); plt.title("WineMLP — Accuracy vs BER (EP, Wald 95% CI)"); plt.legend()
        p3 = os.path.join(args.outdir, "WineMLP_accuracy_vs_BER_EP_Wald.png")
        fig3.tight_layout(); fig3.savefig(p3);
        if args.svg: fig3.savefig(p3.replace(".png",".svg"))
        plt.close(fig3)
    else:
        p3 = None

    # ===== Grafico 4: Wald n = 19000 (punti = FR_wilson) =====
    acc_points_w19000, acc_low_w19000, acc_high_w19000 = [], [], []
    for p in df["FR_wilson"].values:
        lo, hi = wald_ci(float(p), n=args.n19000, z=args.z)
        acc_points_w19000.append((1.0 - p) * 100.0)
        acc_low_w19000.append((1.0 - hi) * 100.0)
        acc_high_w19000.append((1.0 - lo) * 100.0)
    acc_points_w19000 = np.array(acc_points_w19000)
    yerr_w19000 = _nonneg_yerr(acc_points_w19000, np.array(acc_low_w19000), np.array(acc_high_w19000))

    fig4 = plt.figure(figsize=(7, 5), dpi=140)
    plot_with_errors(x, acc_points_w19000, yerr_w19000, marker="d", err_color=COL_WALD_19000,
                     label=f"Wald (n = {args.n19000})")
    _style_axes(); plt.title(f"WineMLP — Accuracy vs BER (Wald 95% CI, n = {args.n19000})"); plt.legend()
    p4 = os.path.join(args.outdir, f"WineMLP_accuracy_vs_BER_Wald_n{args.n19000}.png")
    fig4.tight_layout(); fig4.savefig(p4);
    if args.svg: fig4.savefig(p4.replace(".png",".svg"))
    plt.close(fig4)

    # ===== Grafico 5: One-Step FPC (punti = FR_one_step; barre = ±epsilon tabella) =====
    if (len(N_FPC_LIST) == len(df) == len(ONESTEP_FR_LIST) == len(ONESTEP_EPS_LIST)):
        fr_onestep  = np.array(ONESTEP_FR_LIST, dtype=float)
        eps_onestep = np.array(ONESTEP_EPS_LIST, dtype=float)

        # Limiti FR con epsilon, clamp in [0,1] per sicurezza
        lo_fr = np.clip(fr_onestep - eps_onestep, 0.0, 1.0)
        hi_fr = np.clip(fr_onestep + eps_onestep, 0.0, 1.0)

        # Converti in Accuracy
        acc_points_fpc = (1.0 - fr_onestep) * 100.0
        acc_low_fpc    = (1.0 - hi_fr)      * 100.0
        acc_high_fpc   = (1.0 - lo_fr)      * 100.0
        yerr_fpc = _nonneg_yerr(acc_points_fpc, acc_low_fpc, acc_high_fpc)

        fig5 = plt.figure(figsize=(7, 5), dpi=140)
        plot_with_errors(x, acc_points_fpc, yerr_fpc, marker="^",
                         err_color=COL_WALD_FPC, label="One-Step (FPC, ε tabella)")
        _style_axes(); plt.title("WineMLP — Accuracy vs BER (One-Step FPC, 95%, ε tabella)"); plt.legend()
        p5 = os.path.join(args.outdir, "WineMLP_accuracy_vs_BER_Wald_nFPC.png")
        fig5.tight_layout(); fig5.savefig(p5);
        if args.svg: fig5.savefig(p5.replace(".png",".svg"))
        plt.close(fig5)
    else:
        p5 = None
        print(f"[WARN] Liste FPC non allineate: N_FPC={len(N_FPC_LIST)}, df={len(df)}, "
              f"FR={len(ONESTEP_FR_LIST)}, EPS={len(ONESTEP_EPS_LIST)}")

    # ---------- Log ----------
    print("[OK] Salvato:")
    print(" -", p1)
    if p1b: print(" -", p1b)
    print(" -", p2)
    if p3: print(" -", p3)
    print(" -", p4)
    if p5: print(" -", p5)

if __name__ == "__main__":
    main()
