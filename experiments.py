#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Confronto N≈K (flip globale dei pesi) su DryBean/BeanMLP e Wine/WineMLP.
Esegue due modalità:
  - bitwise-not: ~ su TUTTI gli 8 bit dei pesi quantizzati (equivale a v' = ~v)
  - qnegate    : negazione vera dei pesi nel dominio quantizzato (v' = -v)

Output:
  - CSV risultati (FRcrit, distribuzioni baseline/faulty, K siti, ecc.)
  - CSV qparams (per layer: dtype, qscheme, shape, scale/zero_point)
  - CSV weight stats (per layer e globale: istogramma int_repr a 256 bin, mean/std int & float)

Uso tipico:
  python compare_k_equals_m.py --out_results results.csv --out_qparams qparams.csv --out_weight_stats weight_stats.csv
"""

import os
import csv
import json
import argparse
from itertools import product

import numpy as np
import torch
from tqdm import tqdm

# --- dipendenze dal tuo progetto ---
from faultManager.WeightFault import WeightFault
from faultManager.WeightFaultInjector import WeightFaultInjector
from utils import get_loader, load_from_dict, get_network
import SETTINGS

import matplotlib
matplotlib.use("Agg")  # backend headless
import matplotlib.pyplot as plt



# ============================= Logger =============================

class DualLogger:
    def __init__(self, save_path: str | None):
        self.save_path = save_path
        self.fh = None
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            self.fh = open(save_path, "w", encoding="utf-8")

    def log(self, msg: str):
        print(msg)
        if self.fh:
            self.fh.write(msg + "\n")
            self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close()
            self.fh = None


# ============================= Helper pesi quantizzati & siti =============================

def _get_quant_weight(module: torch.nn.Module) -> torch.Tensor:
    """
    Ritorna il tensore dei pesi *quantizzati* per moduli quantized di PyTorch
    (usa API pubblica e un fallback sul packed).
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

def _enumerate_all_sites(model: torch.nn.Module):
    """
    Enumera tutti i *siti* di peso quantizzato su cui si possono fare flip di bit.
    Yield: (layer_name, tensor_index) per ogni elemento del tensore pesi.
    """
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.quantized.Linear, torch.nn.quantized.Conv2d)):
            try:
                w = _get_quant_weight(module)
            except Exception:
                continue
            shape = w.shape
            for idx in product(*[range(s) for s in shape]):
                yield (name, idx)

def _get_all_quant_modules(model):
    mods = []
    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.quantized.Linear, torch.nn.quantized.Conv2d)):
            try:
                _ = _get_quant_weight(m)
                mods.append((name, m))
            except Exception:
                continue
    return mods


# ============================= Build, quantize, baseline =============================

def build_and_quantize(logger: DualLogger):
    # Forza backend quantizzato CPU
    torch.backends.quantized.engine = "fbgemm"
    device = torch.device("cpu")

    logger.log(f"Loading {SETTINGS.NETWORK_NAME} for {SETTINGS.DATASET_NAME} dataset...")
    model = get_network(SETTINGS.NETWORK_NAME, device, SETTINGS.DATASET_NAME)
    model.to(device).eval()
    ckpt = f"./trained_models/{SETTINGS.DATASET_NAME}_{SETTINGS.NETWORK_NAME}_trained.pth"
    if os.path.exists(ckpt):
        load_from_dict(model, device, ckpt)
        logger.log("[INFO] checkpoint caricato")
    else:
        logger.log(f"[WARN] checkpoint non trovato: {ckpt} (proseguo senza)")

    # Loader
    train_loader, _, test_loader = get_loader(
        dataset_name=SETTINGS.DATASET_NAME,
        batch_size=SETTINGS.BATCH_SIZE,
        network_name=SETTINGS.NETWORK_NAME
    )

    # Quantizzazione se supportata
    if hasattr(model, "quantize_model"):
        model.quantize_model(calib_loader=train_loader)
        model.eval()
        logger.log("[INFO] quantizzazione completata")
    else:
        logger.log("[WARN] modello non quantizzabile: procedo senza quantizzazione")

    # Baseline clean predictions (salvate per-batch per FRcrit)
    clean_by_batch = []
    with torch.inference_mode():
        for xb, _ in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            clean_by_batch.append(torch.argmax(logits, dim=1).cpu())

    # Distribuzione baseline
    clean_flat = torch.cat(clean_by_batch, dim=0) if len(clean_by_batch) > 0 else torch.tensor([], dtype=torch.long)
    num_classes = int(clean_flat.max().item()) + 1 if clean_flat.numel() > 0 else 2
    baseline_hist = np.bincount(clean_flat.numpy(), minlength=num_classes) if clean_flat.numel() > 0 else np.zeros(2, dtype=int)
    baseline_dist = baseline_hist / max(1, baseline_hist.sum())
    logger.log(f"[BASELINE] pred dist = {baseline_dist.tolist()}")

    # Conta siti quantizzati (K)
    all_sites_count = sum(1 for _ in _enumerate_all_sites(model))
    logger.log(f"[INFO] siti quantizzati (K) = {all_sites_count}")

    return model, device, test_loader, clean_by_batch, num_classes, baseline_hist, baseline_dist, all_sites_count


# ============================= Valutazione FRcrit =============================

def eval_frcrit(model, device, test_loader, clean_by_batch, num_classes):
    """
    Calcola FRcrit (= frazione di sample in cui pred_faulty != pred_clean),
    più l'istogramma delle predizioni faulty (per dist).
    """
    mismatches = 0
    total_samples = sum(len(t) for t in clean_by_batch)
    fault_hist = np.zeros(num_classes, dtype=np.int64)

    with torch.inference_mode():
        for batch_i, (xb, _) in enumerate(test_loader):
            xb = xb.to(device)
            logits_f = model(xb)
            pred_f = torch.argmax(logits_f, dim=1).cpu().numpy()

            clean_pred = clean_by_batch[batch_i].numpy()
            mismatches += int((pred_f != clean_pred).sum())
            np.add.at(fault_hist, pred_f, 1)

    frcrit = mismatches / float(total_samples) if total_samples > 0 else 0.0
    return frcrit, fault_hist, total_samples


# ============================= Util per negazione e checks su QTensor =============================

def _negate_qtensor_preserve_qparams(wq: torch.Tensor) -> torch.Tensor:
    """
    Restituisce un QTensor con stessi qparams di wq ma con int_repr negato (two's complement).
    Supporta per-tensor e per-channel. Gestisce qint8 e quint8.
    """
    if not wq.is_quantized:
        raise ValueError("Atteso un QTensor quantizzato.")
    int_repr = wq.int_repr()
    if wq.dtype == torch.qint8:
        int_neg = (-int_repr).to(torch.int8)  # attenzione: -(-128) -> -128
    elif wq.dtype == torch.quint8:
        s = (int_repr.to(torch.int16) - 128).to(torch.int8)
        s_neg = (-s).to(torch.int8)
        int_neg = (s_neg.to(torch.int16) + 128).clamp(0, 255).to(torch.uint8)
    else:
        raise ValueError(f"Quantized dtype non gestito: {wq.dtype}")

    qscheme = wq.qscheme()
    if qscheme in (torch.per_tensor_affine, torch.per_tensor_symmetric):
        scale = wq.q_scale()
        zp = wq.q_zero_point()
        wq_neg = torch._make_per_tensor_quantized_tensor(int_neg, scale, zp)
    elif qscheme in (torch.per_channel_affine, torch.per_channel_symmetric):
        scales = wq.q_per_channel_scales()
        zps = wq.q_per_channel_zero_points()
        axis = wq.q_per_channel_axis()
        wq_neg = torch._make_per_channel_quantized_tensor(int_neg, scales, zps, axis)
    else:
        raise ValueError(f"qscheme non gestito: {qscheme}")
    return wq_neg

def _set_quantized_weight(module: torch.nn.Module, new_wq: torch.Tensor):
    """
    Imposta il peso quantizzato new_wq nel modulo quantized.
    """
    if not (hasattr(module, "_packed_params") and hasattr(module._packed_params, "_weight_bias")):
        raise RuntimeError("Modulo quantizzato senza _packed_params: impossibile sostituire il peso.")
    w_old, b = module._packed_params._weight_bias()
    if hasattr(module._packed_params, "set_weight_bias"):
        module._packed_params.set_weight_bias(new_wq, b)
        return
    # fallback per versioni senza setter
    if isinstance(module, torch.nn.quantized.Linear):
        packed = torch.ops.quantized.linear_prepack(new_wq, b)
        module._packed_params = packed
    elif isinstance(module, torch.nn.quantized.Conv2d):
        stride = module.stride; padding = module.padding; dilation = module.dilation; groups = module.groups
        packed = torch.ops.quantized.conv2d_prepack(new_wq, b, stride, padding, dilation, groups)
        module._packed_params = packed
    else:
        raise RuntimeError("Tipo di modulo quantizzato non supportato per fallback.")


# ============================= QPARAMS (per CSV) =============================

def _qparams_summary(wq: torch.Tensor):
    info = {}
    info["dtype"] = str(wq.dtype).replace("torch.", "")
    info["qscheme"] = str(wq.qscheme()).replace("torch.", "")
    info["shape"] = "x".join(map(str, wq.shape))
    if wq.qscheme() in (torch.per_tensor_affine, torch.per_tensor_symmetric):
        info["scale"] = float(wq.q_scale())
        info["zero_point"] = int(wq.q_zero_point())
        info["scale_min"] = None
        info["scale_max"] = None
        info["axis"] = None
        info["zp_unique"] = None
    else:
        sc = wq.q_per_channel_scales()
        zp = wq.q_per_channel_zero_points()
        info["scale_min"] = float(sc.min())
        info["scale_max"] = float(sc.max())
        info["axis"] = int(wq.q_per_channel_axis())
        zs = sorted(set(int(x) for x in zp.cpu().tolist()))
        info["zp_unique"] = "[" + ",".join(map(str, zs)) + "]"
        info["scale"] = None
        info["zero_point"] = None
    return info

def collect_qparams_rows(model, ds_name, net_name, mode):
    rows = []
    for layer_name, mod in _get_all_quant_modules(model):
        wq = _get_quant_weight(mod)
        s = _qparams_summary(wq)
        rows.append({
            "dataset": ds_name,
            "network": net_name,
            "mode": mode,
            "layer": layer_name,
            "dtype": s["dtype"],
            "qscheme": s["qscheme"],
            "shape": s["shape"],
            "scale": s["scale"],
            "zero_point": s["zero_point"],
            "scale_min": s["scale_min"],
            "scale_max": s["scale_max"],
            "axis": s["axis"],
            "zp_unique": s["zp_unique"],
        })
    return rows


# ============================= (NEW) Weight distributions & stats =============================

def _int_hist_stats_from_qtensor(wq: torch.Tensor):
    """
    Ritorna: (hist_256, n, int_mean, int_std, float_mean, float_std)
    - hist_256: lista di 256 conteggi sull'int_repr (uint8: 0..255; qint8: shift [-128,127]->[0,255]).
    - int_mean/std: sul dominio intero dell'int_repr.
    - float_mean/std: su wq.dequantize() (std non unbiased).
    """
    assert wq.is_quantized, "Atteso QTensor."
    ir = wq.int_repr().cpu().numpy()

    if wq.dtype == torch.qint8:
        # int_repr in [-128,127] -> mappo a [0,255] e FLATTEN
        arr = ir.astype(np.int16, copy=False)
        idx = (arr + 128).astype(np.int64, copy=False).ravel()  # <-- FLATTEN 1-D
    elif wq.dtype == torch.quint8:
        # int_repr già in [0,255] -> FLATTEN
        arr = ir.astype(np.int16, copy=False)
        idx = arr.astype(np.int64, copy=False).ravel()          # <-- FLATTEN 1-D
    else:
        raise ValueError(f"dtype quantizzato non gestito: {wq.dtype}")

    # Ora idx è 1-D di interi non negativi
    hist = np.bincount(idx, minlength=256)
    n = int(idx.size)

    # Statistiche sul dominio intero (prima dello shift per qint8)
    int_mean = float(arr.mean()) if n else 0.0
    int_std  = float(arr.std(ddof=0)) if n else 0.0

    # Statistiche nel dominio float dequantizzato
    dq = wq.dequantize().float().cpu()
    float_mean = float(dq.mean().item()) if n else 0.0
    float_std  = float(dq.std(unbiased=False).item()) if n else 0.0

    return hist.tolist(), n, int_mean, int_std, float_mean, float_std


def collect_weight_stats_rows(model, ds_name, net_name, mode):
    """
    Colleziona, per ogni modulo quantizzato, l'istogramma 256-bin dell'int_repr
    + mean/std int e float. Ritorna lista di righe per CSV e una riga globale __global__.
    """
    rows = []
    global_hist = np.zeros(256, dtype=np.int64)
    g_n = 0
    g_int_sum = 0.0
    g_int_sqsum = 0.0
    g_float_sum = 0.0
    g_float_sqsum = 0.0

    for layer_name, mod in _get_all_quant_modules(model):
        wq = _get_quant_weight(mod)
        h, n, m_i, s_i, m_f, s_f = _int_hist_stats_from_qtensor(wq)
        rows.append({
            "dataset": ds_name, "network": net_name, "mode": mode, "layer": layer_name,
            "dtype": str(wq.dtype).replace("torch.", ""),
            "qscheme": str(wq.qscheme()).replace("torch.", ""),
            "N": n,
            "int_mean": m_i, "int_std": s_i,
            "float_mean": m_f, "float_std": s_f,
            "hist_256": json.dumps(h),
        })
        # aggrega globale (istogramma + momenti)
        global_hist += np.array(h, dtype=np.int64)
        g_n += n
        g_int_sum += m_i * n
        g_int_sqsum += (s_i**2 + m_i**2) * n
        g_float_sum += m_f * n
        g_float_sqsum += (s_f**2 + m_f**2) * n

    if g_n > 0:
        g_int_mean = g_int_sum / g_n
        g_int_var  = max(0.0, g_int_sqsum / g_n - g_int_mean**2)
        g_int_std  = g_int_var**0.5

        g_float_mean = g_float_sum / g_n
        g_float_var  = max(0.0, g_float_sqsum / g_n - g_float_mean**2)
        g_float_std  = g_float_var**0.5
    else:
        g_int_mean = g_int_std = g_float_mean = g_float_std = 0.0

    rows.append({
        "dataset": ds_name, "network": net_name, "mode": mode, "layer": "__global__",
        "dtype": "-", "qscheme": "-", "N": g_n,
        "int_mean": g_int_mean, "int_std": g_int_std,
        "float_mean": g_float_mean, "float_std": g_float_std,
        "hist_256": json.dumps(global_hist.tolist()),
    })
    return rows

def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)

def _plot_overlay(hists_by_mode: dict, title: str, outfile: str):
    """
    hists_by_mode: {mode: np.array shape (256,)} già normalizzati o conteggi
    Plotta overlay (line plot) su 256 bin.
    """
    x = np.arange(256)
    plt.figure(figsize=(9, 4))
    for mode, h in hists_by_mode.items():
        h = np.asarray(h, dtype=np.float64)
        s = h.sum()
        y = h / s if s > 0 else h
        plt.plot(x, y, label=mode)
    plt.title(title)
    plt.xlabel("int_repr (0..255)")
    plt.ylabel("density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

def _plot_global_std_bars(std_by_mode: dict, title: str, outfile: str):
    modes = sorted(std_by_mode.keys())
    vals = [std_by_mode[m] for m in modes]
    plt.figure(figsize=(5, 4))
    plt.bar(modes, vals)
    plt.title(title)
    plt.ylabel("float_std")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

def _make_plots_from_rows(weight_rows_all: list, outdir: str):
    """
    Usa le righe già presenti in memoria (weight_rows_all) per:
      - overlay istogrammi per layer (incl. __global__)
      - bar chart float_std globali
    """
    # indicizza: (dataset, network, layer) -> {mode: (hist, float_std)}
    table = {}
    for r in weight_rows_all:
        ds = r["dataset"]; net = r["network"]; mode = r["mode"]; layer = r["layer"]
        hist = r["hist_256"]
        # se è già list (in memoria), ok; se è string JSON (es. ricarico), fai loads
        if isinstance(hist, str):
            hist = json.loads(hist)
        key = (ds, net, layer)
        table.setdefault(key, {})[mode] = (hist, float(r["float_std"]))

    # plot per-layer (overlay) e bar std globali
    done_std_groups = set()
    for (ds, net, layer), mm in table.items():
        # overlay istogrammi
        hists = {m: mm[m][0] for m in sorted(mm)}
        title = f"{ds}/{net} — layer={layer}"
        fname = f"{_safe_name(ds)}_{_safe_name(net)}_layer-{_safe_name(layer)}.png"
        _plot_overlay(hists, title, os.path.join(outdir, fname))

        # bar chart solo per __global__
        if layer == "__global__":
            stds = {m: mm[m][1] for m in sorted(mm)}
            gkey = (ds, net)
            if gkey not in done_std_groups:
                fname_std = f"{_safe_name(ds)}_{_safe_name(net)}_global_float_std.png"
                _plot_global_std_bars(stds, f"{ds}/{net} — global float_std", os.path.join(outdir, fname_std))
                done_std_groups.add(gkey)

# ============================= Esperimenti: bitwise NOT e qnegate =============================

def run_bitwise_not_all_weights(model, device, test_loader, clean_by_batch, num_classes, logger: DualLogger):
    """
    Usa WeightFaultInjector per fare un'unica injection che fa bitwise-NOT (~)
    di *tutti* i pesi quantizzati (equivalente a flip di tutti gli 8 bit).
    """
    injector = WeightFaultInjector(model)

    # lista di siti quantizzati
    all_sites = list(_enumerate_all_sites(model))
    if len(all_sites) == 0:
        raise RuntimeError("Nessun sito quantizzato trovato: il modello potrebbe non essere quantizzato.")
    faults = [
        WeightFault(injection=1, layer_name=ln, tensor_index=idx, bits=list(range(8)))
        for (ln, idx) in all_sites
    ]

    try:
        injector.inject_faults(faults, 'bit-flip')

        # (NEW) Statistiche pesi dopo NOT
        wrows_fault = collect_weight_stats_rows(model, SETTINGS.DATASET_NAME, SETTINGS.NETWORK_NAME, mode="bitwise-not")
        # log std globale
        for r in wrows_fault:
            if r["layer"] == "__global__":
                logger.log(f"[WEIGHTS] bitwise-not float_std = {r['float_std']:.6g}  (N={r['N']})")
                break

        frcrit, fault_hist, total_samples = eval_frcrit(model, device, test_loader, clean_by_batch, num_classes)
    finally:
        injector.restore_golden()

    return frcrit, fault_hist, total_samples, wrows_fault

def run_true_negate_quantized_all_weights(model, device, test_loader, clean_by_batch, num_classes, logger: DualLogger):
    """
    Negazione *vera* (-q) dei pesi quantizzati su TUTTI i moduli quantized.
    """
    mods = _get_all_quant_modules(model)
    if not mods:
        raise RuntimeError("Nessun modulo quantizzato trovato per negazione vera.")

    # Snapshot prima
    snaps_before = {name: _get_quant_weight(m).clone() for name, m in mods}

    try:
        # applica negazione a livello di int_repr preservando i qparams
        for name, module in mods:
            wq_old = snaps_before[name]
            wq_neg = _negate_qtensor_preserve_qparams(wq_old)
            _set_quantized_weight(module, wq_neg)

        # (NEW) Statistiche pesi dopo negazione vera
        wrows_fault = collect_weight_stats_rows(model, SETTINGS.DATASET_NAME, SETTINGS.NETWORK_NAME, mode="qnegate")
        for r in wrows_fault:
            if r["layer"] == "__global__":
                logger.log(f"[WEIGHTS] qnegate float_std = {r['float_std']:.6g}  (N={r['N']})")
                break

        frcrit, fault_hist, total_samples = eval_frcrit(model, device, test_loader, clean_by_batch, num_classes)
    finally:
        # ripristina golden
        for name, module in mods:
            _set_quantized_weight(module, snaps_before[name])

    return frcrit, fault_hist, total_samples, wrows_fault

def _rebin_hist256(hist, factor=4):
    """Rebin di un istogramma 256-bin in blocchi di 'factor' (divisore di 256)."""
    hist = np.asarray(hist, dtype=np.int64)
    assert 256 % factor == 0, "factor deve dividere 256"
    return hist.reshape(256 // factor, factor).sum(axis=1)

def _plot_int8_hist_bars(h_golden, h_flipped, title: str, outfile: str, rebin=4):
    """
    Istogramma INT8 (-128..127) 'Golden vs Flipped' (globale).
    Normalizzato a densità (area=1). Barre semi-trasparenti.
    """
    # Rebin (meno barre = più leggibile). factor=4 -> 64 barre.
    hg = _rebin_hist256(h_golden, rebin)
    hf = _rebin_hist256(h_flipped, rebin)
    B = len(hg)
    width = rebin  # ampiezza in "unità INT8"
    # Centri dei bin su scala INT8
    centers = (np.arange(B) * rebin - 128) + (rebin / 2.0)

    # Densità (area = 1): dividi anche per la larghezza
    dg = hg / (hg.sum() * width) if hg.sum() > 0 else hg
    df = hf / (hf.sum() * width) if hf.sum() > 0 else hf

    plt.figure(figsize=(9, 4))
    plt.bar(centers, dg, width=width, alpha=0.6, label="Golden", align="center")
    plt.bar(centers, df, width=width, alpha=0.6, label="Flipped", align="center")
    plt.title(title)
    plt.xlabel("INT8 Value")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

def _make_global_int8_barplots(weight_rows_all: list, outdir: str, rebin=4):
    """
    Crea 1 grafico per modalità (bitwise-not e/o qnegate): Golden vs Flipped (layer=__global__).
    """
    # indicizza righe __global__
    globals_by_key = {}  # (ds, net, mode) -> hist_256
    for r in weight_rows_all:
        if r["layer"] != "__global__":
            continue
        hist = r["hist_256"]
        if isinstance(hist, str):
            hist = json.loads(hist)
        globals_by_key[(r["dataset"], r["network"], r["mode"])] = hist

    # per ciascuna coppia (ds,net) crea Golden vs Flipped per ogni mode presente
    modes = ["bitwise-not", "qnegate"]
    pairs = {(ds, net) for (ds, net, _) in globals_by_key.keys()}
    for ds, net in sorted(pairs):
        base = globals_by_key.get((ds, net, "baseline"))
        if base is None:
            continue
        for mode in modes:
            flip = globals_by_key.get((ds, net, mode))
            if flip is None:
                continue
            fname = f"{ds}_{net}_global_int8_{mode}.png".replace(" ", "_")
            title = f"INT8 Quantized Weight Distributions (All Layers) — {ds}/{net} [{mode}]"
            _plot_int8_hist_bars(base, flip, title, os.path.join(outdir, fname), rebin=rebin)

# ============================= Main runner =============================

# Adatta qui i nomi se nel tuo repo sono diversi
DATASETS = [
    ("DryBean", "BeanMLP"),
    ("Wine", "WineMLP"),
]

def run_one(ds_name, net_name, mode, logdir, logger_external=None):
    SETTINGS.DATASET_NAME = ds_name
    SETTINGS.NETWORK_NAME = net_name

    os.makedirs(logdir, exist_ok=True)
    log_path = os.path.join(logdir, f"{ds_name}_{net_name}_{mode}.log")
    logger = logger_external or DualLogger(log_path)

    try:
        model, device, test_loader, clean_by_batch, num_classes, base_hist, base_dist, K_sites = build_and_quantize(logger)
        # QPARAMS (prima dell'injection)
        qparams_rows = collect_qparams_rows(model, ds_name, net_name, mode="baseline")

        # (NEW) Stats pesi baseline
        baseline_wrows = collect_weight_stats_rows(model, ds_name, net_name, mode="baseline")
        # Stampa la std globale baseline (float)
        for r in baseline_wrows:
            if r["layer"] == "__global__":
                logger.log(f"[WEIGHTS] baseline float_std = {r['float_std']:.6g}  (N={r['N']})")
                break

        if mode == "bitwise-not":
            frcrit, fault_hist, total_samples, wrows_fault = run_bitwise_not_all_weights(model, device, test_loader, clean_by_batch, num_classes, logger)
        elif mode == "qnegate":
            frcrit, fault_hist, total_samples, wrows_fault = run_true_negate_quantized_all_weights(model, device, test_loader, clean_by_batch, num_classes, logger)
        else:
            raise ValueError("Mode non valida")

        total_preds = int(fault_hist.sum())
        fault_dist = (fault_hist / total_preds).tolist() if total_preds > 0 else [0.0] * len(base_hist)

        result_row = {
            "dataset": ds_name,
            "network": net_name,
            "mode": mode,
            "K_sites": K_sites,
            "total_samples": total_samples,
            "frcrit": frcrit,
            "baseline_pred_dist": json.dumps((base_hist / max(1, base_hist.sum())).tolist()),
            "fault_pred_dist": json.dumps(fault_dist),
            "logfile": log_path,
        }

        # (NEW) Righe pesi da ritornare (baseline + fault)
        weight_rows = baseline_wrows + wrows_fault

        return result_row, qparams_rows, log_path, weight_rows
    finally:
        if logger_external is None:
            logger.close()

def main():
    ap = argparse.ArgumentParser("Confronto Wine vs DryBean (N≈K): bitwise-not e qnegate con export CSV.")
    ap.add_argument("--modes", nargs="+", default=["bitwise-not", "qnegate"],
                    choices=["bitwise-not", "qnegate"], help="Quali trasformazioni provare")
    ap.add_argument("--out_results", default="compare_results.csv", help="CSV di output con i risultati")
    ap.add_argument("--out_qparams", default="compare_qparams.csv", help="CSV di output con i qparams per layer")
    ap.add_argument("--out_weight_stats", default="weight_stats.csv",
                    help="CSV con istogrammi e statistiche dei pesi (baseline e fault).")
    ap.add_argument("--logdir", default="logs", help="Cartella dei log")
    ap.add_argument("--plots", action="store_true",
                help="Se passato, salva i grafici degli istogrammi per layer e delle std globali.")
    ap.add_argument("--plotdir", default="plots", help="Cartella in cui salvare i plot")
    
    ap.add_argument("--int8bars", action="store_true",
                    help="Salva istogrammi INT8 globali (Golden vs Flipped) per ogni modalità.")
    ap.add_argument("--int8bars-rebin", type=int, default=4,
                    help="Fattore di rebin dei 256 bin (es. 4 -> 64 barre).")


    # opzionale: override datasets
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Override: lista come DryBean:BeanMLP Wine:WineMLP ...")
    args = ap.parse_args()

    # prepara lista dataset/network
    if args.datasets:
        ds_pairs = []
        for tok in args.datasets:
            if ":" not in tok:
                raise SystemExit("Usa formato DATASET:NETWORK (es. DryBean:BeanMLP)")
            ds, net = tok.split(":", 1)
            ds_pairs.append((ds, net))
    else:
        ds_pairs = DATASETS

    # esegui
    results_rows = []
    qparams_rows_all = []
    weight_rows_all = []

    for ds, net in ds_pairs:
        for mode in args.modes:
            # ogni run ha un suo logger file
            log_path = os.path.join(args.logdir, f"{ds}_{net}_{mode}.log")
            logger = DualLogger(log_path)
            try:
                res_row, qrows, _, wrows = run_one(ds, net, mode, logdir=args.logdir, logger_external=logger)
                results_rows.append(res_row)

                # i qparams “baseline” sono identici per le due modalità, salvali una volta per ds/net
                already_q = any((r["dataset"] == ds and r["network"] == net) for r in qparams_rows_all)
                if not already_q:
                    qparams_rows_all.extend(qrows)

                # (NEW) evita duplicati baseline dei pesi quando si eseguono più modalità
                baseline_already = any(
                    (r["dataset"] == ds and r["network"] == net and r["mode"] == "baseline")
                    for r in weight_rows_all
                )
                if baseline_already:
                    wrows = [r for r in wrows if r.get("mode") != "baseline"]

                weight_rows_all.extend(wrows)
            finally:
                logger.close()

    # scrivi CSV risultati
    os.makedirs(os.path.dirname(os.path.abspath(args.out_results)) or ".", exist_ok=True)
    with open(args.out_results, "w", newline="", encoding="utf-8") as fh:
        fieldnames = ["dataset", "network", "mode", "K_sites", "total_samples",
                      "frcrit", "baseline_pred_dist", "fault_pred_dist", "logfile"]
        wr = csv.DictWriter(fh, fieldnames=fieldnames)
        wr.writeheader()
        for r in results_rows:
            wr.writerow(r)

    # scrivi CSV qparams
    os.makedirs(os.path.dirname(os.path.abspath(args.out_qparams)) or ".", exist_ok=True)
    with open(args.out_qparams, "w", newline="", encoding="utf-8") as fh:
        fieldnames = ["dataset", "network", "mode", "layer", "dtype", "qscheme", "shape",
                      "scale", "zero_point", "scale_min", "scale_max", "axis", "zp_unique"]
        wr = csv.DictWriter(fh, fieldnames=fieldnames)
        wr.writeheader()
        for r in qparams_rows_all:
            wr.writerow(r)

    # (NEW) scrivi CSV pesi
    os.makedirs(os.path.dirname(os.path.abspath(args.out_weight_stats)) or ".", exist_ok=True)
    with open(args.out_weight_stats, "w", newline="", encoding="utf-8") as fh:
        fieldnames = ["dataset","network","mode","layer","dtype","qscheme","N",
                      "int_mean","int_std","float_mean","float_std","hist_256"]
        wr = csv.DictWriter(fh, fieldnames=fieldnames)
        wr.writeheader()
        for r in weight_rows_all:
            wr.writerow(r)

    if args.plots:
        os.makedirs(args.plotdir, exist_ok=True)
        _make_plots_from_rows(weight_rows_all, args.plotdir)
        print(f"[PLOTS] Salvati in: {args.plotdir}")

    if args.int8bars:
        os.makedirs(args.plotdir, exist_ok=True)
        _make_global_int8_barplots(weight_rows_all, args.plotdir, rebin=args.int8bars_rebin)
        print(f"[PLOTS-INT8] Salvati in: {args.plotdir}")


    print(f"[DONE] Salvati:\n"
          f"  - risultati: {args.out_results}\n"
          f"  - qparams:   {args.out_qparams}\n"
          f"  - pesi:      {args.out_weight_stats}")

if __name__ == "__main__":
    main()
