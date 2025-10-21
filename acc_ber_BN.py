#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Banknote — Accuracy vs BER con barre di errore (SOLO dati da tabella).
Grafici:
 1) Wilson 95% con n = 1 (didattico, p̂ = FR_wilson)
 1b) Wald 95% con n = 1 (didattico, p̂ = FR_ep)
 2) Wilson 95% (punti = FR_wilson, barre = [wilson min, wilson max])
 3) EP (punti = FR_ep, barre = [wald min, wald max])
 4) Wald 95% con n = 19000 costante (punti = FR_wilson)
Salva in plots/acc_ber_banknote/.
"""

import os
import math
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- Parametri ----------------
M_DEFAULT = 288  # Total Faults per Banknote

# ---------------- Tabella (copiata 1:1) ----------------
# Colonne:
# K, N, FR_exhaustive, n_ep, FR_ep, eps_ep, wald_min, wald_max,
# n_wilson, FR_wilson, eps_wilson, wilson_min, wilson_max

# CAMBIARE 287	288			9124	0.06338559	0.00496841	0.05838594 !!! ERA SBAGLIATO
ROWS = [
    (1,    288,          0.00369134,  634,   0.00408487, 0.00496491, 0.0,         0.00904978,  800,   0.00401092, 0.00497093, 0.00141035, 0.01135221),
    (2,    41328,        0.00695132,  1069,  0.00697276, 0.00498827, 0.00198449,  0.01196104,  1200,  0.00699029, 0.00496246, 0.00360108, 0.01352601),
    (3,    3939936,      0.00993804,  1500,  0.00983819, 0.00499483, 0.00484335,  0.01483302,  1600,  0.00982403, 0.00496773, 0.00603039, 0.01596585),
    (4,    280720440,    float("nan"),1889,  0.01243274, 0.00499697, 0.00743576,  0.01742971,  2000,  0.01234102, 0.00492354, 0.00835238, 0.01819946),
    (5,    15944920992,  float("nan"),2352,  0.01553212, 0.00499751, 0.01053461,  0.02052963,  2400,  0.01545105, 0.00499105, 0.01123436, 0.02121646),
    (6,    7.52069e11,   float("nan"),2738,  0.01813031, 0.00499768, 0.01313262,  0.02312799,  2800,  0.01804785, 0.00497167, 0.01373651, 0.02367985),
    (7,    3.02976e13,   float("nan"),3134,  0.02067444, 0.00498181, 0.01569263,  0.02565625,  3150,  0.02062876, 0.00499498, 0.01621768, 0.02620765),
    (8,    1.0642e15,    float("nan"),3550,  0.02366060, 0.00499983, 0.01866077,  0.02866043,  3600,  0.02358617, 0.00498068, 0.01911333, 0.02907468),
    (9,    3.31086e16,   float("nan"),3900,  0.02601817, 0.00499617, 0.02102200,  0.03101435,  3950,  0.02601942, 0.00498348, 0.02149646, 0.03146342),
    (10,   9.23729e17,   float("nan"),4348,  0.02908825, 0.00499528, 0.02409297,  0.03408354,  4400,  0.02907381, 0.00497929, 0.02450532, 0.03446390),
    (50,   3.33659e56,   float("nan"),22847, 0.18143265, 0.00499720, 0.17643545,  0.18642985,  22850, 0.18148690, 0.00499732, 0.17654313, 0.18653776),
    (100,  2.86062e79,   float("nan"),37224, 0.41118770, 0.00499865, 0.40618905,  0.41618635,  37250, 0.41120336, 0.00499669, 0.40621583, 0.41620921),
    (144,  2.33618e85,   float("nan"),38412, 0.50741532, 0.00499971, 0.50241561,  0.51241503,  38450, 0.50743697, 0.00499699, 0.50243924, 0.51243321),
    (150,  1.82086e85,   float("nan"),38408, 0.50759159, 0.00499994, 0.50259164,  0.51259153,  38450, 0.50761435, 0.00499696, 0.50261663, 0.51261055),
    (200,  4.91978e75,   float("nan"),36900, 0.40004552, 0.00499870, 0.39504682,  0.40504421,  36900, 0.40009209, 0.00499854, 0.39510394, 0.40510103),
    (287,  288,          float("nan"),5800,  0.03877260, 0.00496841, 0.05838594,  0.06838523,  9150,  0.06350204, 0.00499912, 0.05868611, 0.06868435),
    (288,  1,            float("nan"),8759,  0.06067961, 0.00499985, 0.05567976,  0.06567946,  8800,  0.06067961, 0.00499078, 0.05588053, 0.06586210),
]

COLUMNS = [
    "K","N","FR_exhaustive",
    "n_ep","FR_ep","eps_ep","wald_min","wald_max",
    "n_wilson","FR_wilson","eps_wilson","wilson_min","wilson_max",
]

# ---------------- Statistiche ----------------
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

# ---------------- Stile & helper ----------------
def _style_axes():
    plt.xlabel("BER = K / M")
    plt.ylabel("Accuracy (1 − FR) [%]")
    plt.ylim(0, 100)
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    plt.gca().set_axisbelow(True)

LINE_BASE = dict(color="k", linewidth=1.2, markeredgecolor="k",
                 markerfacecolor="white", markeredgewidth=1.2, zorder=2)

def plot_with_errors(x, y, yerr, marker, err_color, label):
    line_kw = LINE_BASE.copy()
    line_kw["marker"] = marker
    plt.plot(x, y, **line_kw, label=label)
    plt.errorbar(x, y, yerr=yerr, fmt="none",
                 ecolor=err_color, elinewidth=2.2, capsize=6, capthick=2.2,
                 alpha=0.98, zorder=3)

# Colori barre
COL_WIL_N1     = "#1f77b4"  # blu
COL_WALD_N1_EP = "#17becf"  # ciano
COL_WIL_TABLE  = "#2ca02c"  # verde
COL_EP_WALD    = "#d62728"  # rosso
COL_WALD_19000 = "#9467bd"  # viola

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--M", type=int, default=M_DEFAULT)
    ap.add_argument("--outdir", type=str, default="plots/acc_ber_banknote")
    ap.add_argument("--svg", action="store_true")
    ap.add_argument("--z", type=float, default=1.96)
    ap.add_argument("--n19000", type=int, default=19000)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.DataFrame(ROWS, columns=COLUMNS).sort_values("K").reset_index(drop=True)

    # BER
    df["BER"] = df["K"] / float(args.M)

    # Avvisi su possibili incoerenze tabellari (es. K=287)
    bad = df[(df["wald_min"] > df["wald_max"]) | (df["wald_min"] > df["FR_ep"] + df["eps_ep"] + 1e-6)]
    if len(bad):
        ks = ", ".join(str(int(k)) for k in bad["K"].tolist())
        print(f"[WARN] EP wald_min/max incoerenti alle righe K={ks}: verranno comunque usati come da tabella.")

    # ---------- Grafico 1: Wilson 95% con n = 1 ----------
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
    _style_axes(); plt.title("Banknote — Accuracy vs BER (Wilson 95% CI, n = 1)"); plt.legend()
    p1 = os.path.join(args.outdir, "Banknote_accuracy_vs_BER_Wilson_n1.png")
    fig1.tight_layout(); fig1.savefig(p1)
    if args.svg: fig1.savefig(p1.replace(".png",".svg"))
    plt.close(fig1)

    # ---------- Grafico 1b: Wald 95% con n = 1 (p̂ = FR_ep) ----------
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
    _style_axes(); plt.title("Banknote — Accuracy vs BER (Wald 95% CI, n = 1, p̂ = EP)"); plt.legend()
    p1b = os.path.join(args.outdir, "Banknote_accuracy_vs_BER_Wald_n1_EP.png")
    fig1b.tight_layout(); fig1b.savefig(p1b)
    if args.svg: fig1b.savefig(p1b.replace(".png",".svg"))
    plt.close(fig1b)

    # ---------- Grafico 2: Wilson 95% (tabella) ----------
    acc_points_w = (1.0 - df["FR_wilson"].values) * 100.0
    acc_low_w    = (1.0 - df["wilson_max"].values) * 100.0
    acc_high_w   = (1.0 - df["wilson_min"].values) * 100.0
    yerr_w = _nonneg_yerr(acc_points_w, acc_low_w, acc_high_w)

    fig2 = plt.figure(figsize=(7, 5), dpi=140)
    plot_with_errors(x, acc_points_w, yerr_w, marker="o", err_color=COL_WIL_TABLE, label="Wilson (tabella)")
    _style_axes(); plt.title("Banknote — Accuracy vs BER (Wilson 95% dalla tabella)"); plt.legend()
    p2 = os.path.join(args.outdir, "Banknote_accuracy_vs_BER_Wilson_table.png")
    fig2.tight_layout(); fig2.savefig(p2)
    if args.svg: fig2.savefig(p2.replace(".png",".svg"))
    plt.close(fig2)

    # ---------- Grafico 3: EP (Wald 95% dalla tabella) ----------
    acc_points_ep = (1.0 - df["FR_ep"].values) * 100.0
    acc_low_ep    = (1.0 - df["wald_max"].values) * 100.0
    acc_high_ep   = (1.0 - df["wald_min"].values) * 100.0
    yerr_ep = _nonneg_yerr(acc_points_ep, acc_low_ep, acc_high_ep)

    fig3 = plt.figure(figsize=(7, 5), dpi=140)
    plot_with_errors(x, acc_points_ep, yerr_ep, marker="s", err_color=COL_EP_WALD, label="EP (Wald 95% tabella)")
    _style_axes(); plt.title("Banknote — Accuracy vs BER (EP, Wald 95% CI)"); plt.legend()
    p3 = os.path.join(args.outdir, "Banknote_accuracy_vs_BER_EP_Wald_table.png")
    fig3.tight_layout(); fig3.savefig(p3)
    if args.svg: fig3.savefig(p3.replace(".png",".svg"))
    plt.close(fig3)

    # ---------- Grafico 4: Wald 95% con n = 19000 (punti = FR_wilson) ----------
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
    _style_axes(); plt.title(f"Banknote — Accuracy vs BER (Wald 95% CI, n = {args.n19000})"); plt.legend()
    p4 = os.path.join(args.outdir, f"Banknote_accuracy_vs_BER_Wald_n{args.n19000}.png")
    fig4.tight_layout(); fig4.savefig(p4)
    if args.svg: fig4.savefig(p4.replace(".png",".svg"))
    plt.close(fig4)

    # ---------- Exhaustive (se presente) ----------
    df_exh = df[~df["FR_exhaustive"].isna()].copy()
    if len(df_exh):
        acc_exh = (1.0 - df_exh["FR_exhaustive"].values) * 100.0
        figE = plt.figure(figsize=(7, 5), dpi=140)
        plt.plot(df_exh["BER"].values, acc_exh, linestyle="--", marker="x", color="#1f77b4", label="Exhaustive (punti)")
        _style_axes(); plt.title("Banknote — Accuracy vs BER (Exhaustive)"); plt.legend()
        pE = os.path.join(args.outdir, "Banknote_accuracy_vs_BER_Exhaustive.png")
        figE.tight_layout(); figE.savefig(pE)
        if args.svg: figE.savefig(pE.replace(".png",".svg"))
        plt.close(figE)

    # ---------- Log ----------
    print("[OK] Salvato:")
    for p in [p1, p1b, p2, p3, p4]:
        print(" -", p)

if __name__ == "__main__":
    main()
