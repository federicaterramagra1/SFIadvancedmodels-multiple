#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Accuracy vs BER — Wilson-only (log-scale X), with:
  1) Summary graph from table (Wilson 95% CI)
  2) n = 1 graph (Wilson 95% CI) using one chosen run
  3) n = N graph (Wilson 95% CI), e.g. N = 100 (autogen or CSV)

Notes
- BER = K / M
- K = number of simultaneous bit-flips injected in one shot.
- M = total number of elementary fault sites (all bits of quantized weight tensors considered).

Inputs
- Summary CSV (required): must contain at least the columns
    K, FR_wilson, FR_wilson_low, FR_wilson_high
  (other columns are ignored)

- n=1 CSV (optional unless autogenerating): columns can be either
    (K, run, fr_n1)  or  (K, rep, fr_n1)
  If multiple runs are present, choose which one via --n1-use-run.

- n=N CSV (optional unless autogenerating): columns
    (K, p_hat, n)       # p_hat is the mean FR across n random injections for that K

Autogeneration
- n=1 autogen: use --autogen-n1 to generate (K, run, fr_n1), with --n1-K-list and --n1-seed.
- n=N autogen: use --autogen-nN and --nN to generate N injections per K, computing
  p_hat = mean(FR_i) across N random injections for that K, then Wilson CI with n = N.

Outputs (PNG)
- {title}_accuracy_vs_BER_Wilson_Summary.png
- {title}_accuracy_vs_BER_Wilson_n1_run{r}.png
- {title}_accuracy_vs_BER_Wilson_n{N}.png
"""

import os
import math
import argparse
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================ Style ============================

LINE_BASE = dict(
    color="k", linewidth=1.6,
    markeredgecolor="k", markerfacecolor="white",
    markeredgewidth=1.3, zorder=3
)

COL_WIL_SUM   = "#2ca02c"  # summary Wilson bands
COL_WIL_N1    = "#1f77b4"  # n=1 Wilson bands
COL_WIL_NN    = "#8c564b"  # n=N Wilson bands

def _style_axes_logx(xlim=(1e-4, 1.0)):
    plt.xlabel("BER")
    plt.ylabel("Accuracy (1 − FR) [%]")
    plt.ylim(0, 100)
    ax = plt.gca()
    ax.set_xscale("log")
    if xlim is not None:
        ax.set_xlim(*xlim)  # ora arriva fino a 1.0 (10^0)
    plt.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_axisbelow(True)

def _acc_from_fr(fr):
    return (1.0 - np.asarray(fr, dtype=float)) * 100.0

def _nonneg_yerr(center, low, high):
    center = np.asarray(center, dtype=float)
    low = np.minimum(np.asarray(low, dtype=float), center)
    high = np.maximum(np.asarray(high, dtype=float), center)
    neg = center - low
    pos = high - center
    neg[neg < 0] = 0; pos[pos < 0] = 0
    return np.vstack([neg, pos])

def wilson_ci(p_hat: float, n: int, z: float = 1.96):
    if n <= 0: return 0.0, 1.0
    p = float(min(max(p_hat, 1e-12), 1.0 - 1e-12))
    denom  = 1.0 + (z*z)/n
    center = (p + (z*z)/(2*n)) / denom
    half   = (z * math.sqrt((p*(1.0 - p)/n) + (z*z)/(4*n*n))) / denom
    return max(0.0, center - half), min(1.0, center + half)

def _plot_series_with_bands(x, y_acc, low_acc, high_acc, color, marker, label, title, outpath):
    yerr = _nonneg_yerr(y_acc, low_acc, high_acc)
    fig = plt.figure(figsize=(7.2, 5.2), dpi=140)
    plt.plot(x, y_acc, marker=marker, label=label, **LINE_BASE)
    plt.errorbar(x, y_acc, yerr=yerr, fmt="none",
                 ecolor=color, elinewidth=2.0, capsize=5, capthick=2.0, alpha=0.95, zorder=2)
    _style_axes_logx()
    plt.title(title)
    plt.legend()
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    fig.tight_layout(); fig.savefig(outpath); plt.close(fig)
    return outpath

# ============================ Helpers: column normalization ============================

def _normalize_n1_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize columns to exactly: K, run, fr_n1."""
    mapping = {}
    lowmap = {c.lower(): c for c in df.columns}
    if "k" in lowmap: mapping[lowmap["k"]] = "K"
    if "run" in lowmap: mapping[lowmap["run"]] = "run"
    elif "rep" in lowmap: mapping[lowmap["rep"]] = "run"
    if "fr_n1" in lowmap: mapping[lowmap["fr_n1"]] = "fr_n1"
    elif "fr" in lowmap: mapping[lowmap["fr"]] = "fr_n1"
    elif "failure rate" in lowmap: mapping[lowmap["failure rate"]] = "fr_n1"
    return df.rename(columns=mapping)

# ============================ Table → Summary Wilson ============================

def plot_summary_wilson(table_csv: str, M: int, outdir: str, title: str):
    df = pd.read_csv(table_csv)
    for c in ["K", "FR_wilson", "FR_wilson_low", "FR_wilson_high"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("K").dropna(subset=["K", "FR_wilson"]).reset_index(drop=True)

    x      = (df["K"].astype(float) / float(M)).values
    acc    = _acc_from_fr(df["FR_wilson"].values)
    acc_lo = _acc_from_fr(df["FR_wilson_high"].values)  # FR high -> Acc low
    acc_hi = _acc_from_fr(df["FR_wilson_low"].values)

    out = os.path.join(outdir, f"{title}_accuracy_vs_BER_Wilson_Summary.png")
    return _plot_series_with_bands(
        x, acc, acc_lo, acc_hi, COL_WIL_SUM, "o",
        label="Wilson (summary table)", 
        title=f"{title} — Accuracy vs BER (Wilson 95% CI, summary)",
        outpath=out
    )

# ============================ FI Context (single build, reused) ============================

@dataclass
class FIContext:
    model: "torch.nn.Module"
    device: "torch.device"
    test_loader: "torch.utils.data.DataLoader"
    clean_by_batch: List["torch.Tensor"]
    total_samples: int
    all_faults: List[Tuple[str, Tuple[int, ...], int]]
    M: int

def _get_quant_weight(module):
    import torch
    if hasattr(module, "weight"):
        try:
            return module.weight()
        except Exception:
            pass
    if hasattr(module, "_packed_params") and hasattr(module._packed_params, "_weight_bias"):
        w, _ = module._packed_params._weight_bias()
        return w
    raise RuntimeError("Cannot access quantized weights.")

def _build_all_faults(model):
    from itertools import product
    import torch.nn as nn
    faults = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.quantized.Linear, nn.quantized.Conv2d)):
            try:
                w = _get_quant_weight(module)
            except Exception:
                continue
            for idx in product(*[range(s) for s in w.shape]):
                for bit in range(8):
                    faults.append((name, idx, bit))
    return faults

def build_fi_context() -> FIContext:
    import torch
    from utils import get_loader, load_from_dict, get_network, load_quantized_model, save_quantized_model
    import SETTINGS

    torch.backends.quantized.engine = "fbgemm"
    device = torch.device("cpu")
    ds = getattr(SETTINGS, "DATASET_NAME", getattr(SETTINGS, "DATASET", ""))
    net = getattr(SETTINGS, "NETWORK_NAME", getattr(SETTINGS, "NETWORK", ""))

    print(f"Loading {ds} dataset...")
    train_loader, _, test_loader = get_loader(
        dataset_name=ds,
        batch_size=getattr(SETTINGS, "BATCH_SIZE", 64),
        network_name=net
    )

    qmodel, qpath = load_quantized_model(ds, net, device="cpu", engine="fbgemm")
    if qmodel is not None:
        model = qmodel
        print(f"[PTQ] loaded quantized: {qpath}")
    else:
        model = get_network(net, device, ds).to(device).eval()
        ckpt = f"./trained_models/{ds}_{net}_trained.pth"
        if os.path.exists(ckpt):
            load_from_dict(model, device, ckpt)
            print("[CKPT] float checkpoint loaded")
        if hasattr(model, "quantize_model"):
            model.quantize_model(calib_loader=train_loader)
            model.eval()
            save_quantized_model(model, ds, net, engine="fbgemm")
            print("[PTQ] quantized & saved")

    clean_by_batch = []
    with torch.inference_mode():
        for xb, _ in test_loader:
            logits = model(xb.to(device))
            clean_by_batch.append(torch.argmax(logits, dim=1).cpu())
    total_samples = sum(len(t) for t in clean_by_batch)
    if total_samples == 0:
        raise RuntimeError("Empty test loader.")

    all_faults = _build_all_faults(model)
    M = len(all_faults)
    print(f"[FI] Elementary faults M = {M}")
    return FIContext(model, device, test_loader, clean_by_batch, total_samples, all_faults, M)

# ============================ n=N: CSV or autogen (reusing context) ============================

def autogen_n_points(Ks: Sequence[int], N: int, seed: int, out_csv: Optional[str], ctx: Optional[FIContext] = None) -> pd.DataFrame:
    """
    For each K, sample N random K-combinations, evaluate FR for each injection,
    compute p_hat = mean(FR_i). Save CSV with (K, p_hat, n=N).
    If ctx is provided, reuse the prebuilt model/loader/fault-list.
    """
    import random
    import torch
    from faultManager.WeightFault import WeightFault
    from faultManager.WeightFaultInjector import WeightFaultInjector

    if ctx is None:
        ctx = build_fi_context()

    Ks = list(sorted(set(int(k) for k in Ks)))
    rows = []
    inj_id = 0
    rnd = random.Random(seed)

    for K in Ks:
        if K > ctx.M:
            print(f"[WARN] K={K} > M={ctx.M}: skipping.")
            continue
        vals = []
        for _ in range(N):
            idxs = rnd.sample(range(ctx.M), K)
            combo = [ctx.all_faults[i] for i in idxs]
            injector = WeightFaultInjector(ctx.model)
            faults = [WeightFault(injection=inj_id + 1, layer_name=ln, tensor_index=ti, bits=[bt])
                      for (ln, ti, bt) in combo]
            mismatches = 0
            try:
                injector.inject_faults(faults, 'bit-flip')
                with torch.inference_mode():
                    for batch_i, (xb, _) in enumerate(ctx.test_loader):
                        pred_f = torch.argmax(ctx.model(xb.to(ctx.device)), dim=1).cpu().numpy()
                        clean_pred = ctx.clean_by_batch[batch_i].numpy()
                        mismatches += int((pred_f != clean_pred).sum())
            finally:
                injector.restore_golden()
            inj_id += 1
            vals.append(float(mismatches / float(ctx.total_samples)))
        p_hat = float(np.mean(vals)) if len(vals) else float("nan")
        rows.append((K, p_hat, int(N)))

    df = pd.DataFrame(rows, columns=["K","p_hat","n"]).sort_values("K").reset_index(drop=True)
    if out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)) or ".", exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"[n={N}] saved: {out_csv}")
    return df

def plot_nN_wilson(nN_csv: str, M: int, outdir: str, title: str, z: float):
    df = pd.read_csv(nN_csv)
    for c in ["K","p_hat","n"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["K","p_hat","n"]).sort_values("K").reset_index(drop=True)

    x   = (df["K"].astype(float) / float(M)).values
    p   = df["p_hat"].astype(float).values
    nn  = df["n"].astype(int).values

    lo = []; hi = []
    for pi, n_i in zip(p, nn):
        l, h = wilson_ci(pi, n=int(n_i), z=z)
        lo.append(l); hi.append(h)

    acc    = (1.0 - p)            * 100.0
    acc_lo = (1.0 - np.array(hi)) * 100.0
    acc_hi = (1.0 - np.array(lo)) * 100.0

    Nuniq = sorted(set(nn.tolist()))
    tag = f"n{Nuniq[0]}" if len(Nuniq)==1 else f"n{min(Nuniq)}-{max(Nuniq)}"
    out = os.path.join(outdir, f"{title}_accuracy_vs_BER_Wilson_{tag}.png")
    return _plot_series_with_bands(
        x, acc, acc_lo, acc_hi, COL_WIL_NN, "o",
        label=f"Wilson ({tag})",
        title=f"{title} — Accuracy vs BER (Wilson 95% CI, {tag})",
        outpath=out
    )

# ============================ n=1 plotting ============================

def plot_n1_wilson(n1_csv: str, M: int, outdir: str, title: str, z: float, use_run: int):
    df = pd.read_csv(n1_csv)
    df = _normalize_n1_columns(df)
    for c in ["K","run","fr_n1"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")
    sub = df[df["run"] == use_run].sort_values("K").dropna(subset=["K","fr_n1"])
    if sub.empty:
        raise ValueError(f"No rows for run={use_run} in {n1_csv}")

    x   = (sub["K"].astype(float) / float(M)).values
    p   = sub["fr_n1"].astype(float).values

    lo, hi = [], []
    for pi in p:
        l, h = wilson_ci(pi, n=1, z=z)
        lo.append(l); hi.append(h)

    acc    = (1.0 - p)            * 100.0
    acc_lo = (1.0 - np.array(hi)) * 100.0
    acc_hi = (1.0 - np.array(lo)) * 100.0

    out = os.path.join(outdir, f"{title}_accuracy_vs_BER_Wilson_n1_run{int(use_run)}.png")
    return _plot_series_with_bands(
        x, acc, acc_lo, acc_hi, COL_WIL_N1, "o",
        label=f"run {int(use_run)} (n = 1)",
        title=f"{title} — Accuracy vs BER (Wilson 95% CI, n = 1, run {int(use_run)})",
        outpath=out
    )

# ============================ Main ============================

def main():
    ap = argparse.ArgumentParser()
    # common
    ap.add_argument("--table-csv", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="plots/acc_ber")
    ap.add_argument("--title", type=str, default="Model")
    ap.add_argument("--M", type=int, required=True, help="Total elementary fault sites (M)")
    ap.add_argument("--z", type=float, default=1.96)

    # n=1
    ap.add_argument("--n1-csv", type=str, default=None)
    ap.add_argument("--n1-use-run", type=int, default=1, help="Which run to plot for n=1")
    ap.add_argument("--autogen-n1", action="store_true")
    ap.add_argument("--n1-K-list", type=str, default="")
    ap.add_argument("--n1-runs", type=int, default=1)
    ap.add_argument("--n1-seed", type=int, default=0)
    ap.add_argument("--n1-out", type=str, default="plots/acc_ber/n1_points.csv")

    # n=N
    ap.add_argument("--nN-csv", type=str, default=None)
    ap.add_argument("--autogen-nN", action="store_true")
    ap.add_argument("--nN", type=int, default=100)
    ap.add_argument("--nN-K-list", type=str, default="")
    ap.add_argument("--nN-seed", type=int, default=0)
    ap.add_argument("--nN-out", type=str, default="plots/acc_ber/nN_points.csv")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    saved = []

    # 1) Summary (Wilson-only)
    saved.append(plot_summary_wilson(args.table_csv, M=args.M, outdir=args.outdir, title=args.title))

    # Build context ONCE if any autogen is required
    ctx: Optional[FIContext] = None
    need_ctx = args.autogen_n1 or args.autogen_nN
    if need_ctx:
        ctx = build_fi_context()

    # 2) n = 1 (Wilson-only)
    if args.autogen_n1:
        if not args.n1_K_list:
            raise ValueError("--autogen-n1 requires --n1-K-list (comma-separated).")
        Ks = [int(x.strip()) for x in args.n1_K_list.split(",") if x.strip()]
        # generate runs; save all in one CSV
        all_rows = []
        for run in range(1, args.n1_runs + 1):
            df_run = autogen_n_points(Ks, N=1, seed=args.n1_seed + 10007*run, out_csv=None, ctx=ctx)
            for _, r in df_run.iterrows():
                all_rows.append((int(r["K"]), run, float(r["p_hat"])))
        df_all = pd.DataFrame(all_rows, columns=["K","run","fr_n1"]).sort_values(["run","K"])
        df_all.to_csv(args.n1_out, index=False)
        print(f"[n=1] saved: {args.n1_out}")
        n1_csv_path = args.n1_out
    else:
        if not args.n1_csv:
            raise ValueError("Please provide --n1-csv or enable --autogen-n1.")
        n1_csv_path = args.n1_csv

    saved.append(plot_n1_wilson(n1_csv_path, M=args.M, outdir=args.outdir,
                                title=args.title, z=args.z, use_run=args.n1_use_run))

    # 3) n = N (Wilson-only, e.g. 100)
    if args.autogen_nN:
        if not args.nN_K_list:
            raise ValueError("--autogen-nN requires --nN-K-list (comma-separated).")
        Ks = [int(x.strip()) for x in args.nN_K_list.split(",") if x.strip()]
        autogen_n_points(Ks, N=args.nN, seed=args.nN_seed, out_csv=args.nN_out, ctx=ctx)
        nN_csv_path = args.nN_out
    else:
        if not args.nN_csv:
            raise ValueError("Please provide --nN-csv or enable --autogen-nN.")
        nN_csv_path = args.nN_csv

    saved.append(plot_nN_wilson(nN_csv_path, M=args.M, outdir=args.outdir,
                                title=args.title, z=args.z))

    print("[OK] Saved:")
    for p in saved:
        print(" -", p)

if __name__ == "__main__":
    main()
