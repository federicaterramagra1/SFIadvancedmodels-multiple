#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Exhaustive-FR (FAST) sulla STESSA popolazione di EP/Wilson — tutto-in-uno.

- Costruisce il modello, fa PTQ (se supportato), calcola le golden by-batch.
- Enumera ESAUSTIVAMENTE le combinazioni di K bit-flip sulla stessa popolazione EP/Wilson
  (moduli quantizzati torch.nn.quantized.{Linear,Conv2d}, 8 bit per peso o subset).
- Supporta:
  * Sharding manuale (EXA_SHARDS_TOTAL, EXA_SHARD_IDX)
  * Auto-parallelismo locale multi-processo (EXA_AUTOSPAWN, EXA_PROCS)
  * (Opz.) Sotto-campionamento batch con garanzia di Hoeffding (EXA_TEST_EPS/DELTA) — OFF di default
  * Top-K peggiori injection (diagnostica)
- Salva un file per shard e fa AUTOMATICAMENTE il MERGE pesato finale (FRcrit globale per K).

Env/SETTINGS (default = esaustivo):
  EXA_K_LIST=[1,2,3]  EXA_TARGET_MODULES=[]  EXA_BITS=[0..7]
  EXA_MAX_COMBOS=None  EXA_SHARDS_TOTAL=1  EXA_SHARD_IDX=0
  EXA_AUTOSPAWN=0  EXA_PROCS=os.cpu_count()
  EXA_TEST_EPS=0  EXA_TEST_DELTA=1e-3
  TOP_K=100  SAVE_DIR="results_minimal"

Esempio multicore locale (esaustiva completa, nessun sotto-campionamento):
  EXA_K_LIST=1,2,3 EXA_AUTOSPAWN=1 EXA_PROCS=$(nproc) python exhaustive_samepop_fast.py
"""

import os, re, json
import math, time, heapq
from itertools import product, combinations, islice
from typing import List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm

import SETTINGS
from utils import get_network, get_loader, load_from_dict
from faultManager.WeightFaultInjector import WeightFaultInjector
from faultManager.WeightFault import WeightFault

try:
    import psutil  # per set_num_threads
except Exception:
    psutil = None

torch.backends.quantized.engine = "fbgemm"

# indice dei batch selezionati (per Hoeffding); impostato nel main/worker
SUBSET_IDX: Optional[List[int]] = None


# ---------------- Utils ----------------

def _sci_format_comb(n: int, k: int) -> str:
    if k < 0 or k > n: return "0"
    k = min(k, n - k)
    if k == 0: return "1"
    ln10 = math.log(10.0)
    log10_val = (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / ln10
    exp = int(math.floor(log10_val))
    mant = 10 ** (log10_val - exp)
    return f"{mant:.3f}e+{exp:d}"


def _parse_env_list(val: str):
    parts = [p.strip() for p in val.split(",") if p.strip() != ""]
    ints = []
    for p in parts:
        try: ints.append(int(p))
        except Exception: ints.append(p)
    return ints


def _env_or_settings(name: str, default):
    """env > SETTINGS > default — con parsing automatico di bool/int/list/None."""
    if name in os.environ:
        raw = os.environ[name]
        if isinstance(default, bool):
            return str(raw).lower() in ("1", "true", "yes", "y", "on")
        try:
            return int(raw)
        except Exception:
            pass
        if "," in raw:
            return _parse_env_list(raw)
        if raw.strip().lower() in ("none", "null", ""):
            return None
        return raw
    if hasattr(SETTINGS, name):
        return getattr(SETTINGS, name)
    return default


# ---------------- Popolazione fault = stessa degli script EP/Wilson ----------------

def _get_quant_weight(module):
    if hasattr(module, "weight"):
        try: return module.weight()
        except Exception: pass
    if hasattr(module, "_packed_params") and hasattr(module._packed_params, "_weight_bias"):
        w, _ = module._packed_params._weight_bias()
        return w
    raise RuntimeError("Impossibile ottenere il peso quantizzato dal modulo.")


def _build_all_faults(model,
                      target_modules: Optional[List[str]] = None,
                      bits: Optional[List[int]] = None,
                      as_list=True):
    """
    Solo torch.nn.quantized.{Linear, Conv2d}, bit 0..7 (o subset).
    Filtro opzionale su prefisso 'root' del nome modulo (EXA_TARGET_MODULES).
    """
    tmods = set(target_modules or [])
    use_filter = len(tmods) > 0
    bits = list(bits) if bits not in (None, []) else list(range(8))

    def _iter():
        for name, module in model.named_modules():
            if use_filter:
                root = name.split(".")[0]
                if root not in tmods:
                    continue
            if isinstance(module, (torch.nn.quantized.Linear, torch.nn.quantized.Conv2d)):
                try:
                    w = _get_quant_weight(module)
                except Exception:
                    continue
                shape = w.shape
                for idx in product(*[range(s) for s in shape]):
                    for bit in bits:
                        yield (name, idx, bit)

    return list(_iter()) if as_list else _iter()


# ---------------- Build + PTQ + golden-by-batch ----------------

def _get_ds_net_from_settings():
    dataset = getattr(SETTINGS, "DATASET_NAME", None) or getattr(SETTINGS, "DATASET", None)
    network = getattr(SETTINGS, "NETWORK_NAME", None) or getattr(SETTINGS, "NETWORK", None)
    if not dataset or not network:
        raise RuntimeError("SETTINGS: specifica DATASET_NAME/DATASET e NETWORK_NAME/NETWORK.")
    return dataset, network


def build_and_quantize_once():
    """Rete su CPU, ckpt, PTQ (se disponibile), test_batches list, golden by-batch, baseline."""
    device = torch.device("cpu")
    dataset, network = _get_ds_net_from_settings()

    model = get_network(network, device, dataset).to(device).eval()

    ckpt = f"./trained_models/{dataset}_{network}_trained.pth"
    if os.path.exists(ckpt):
        load_from_dict(model, device, ckpt)
        print(f"[CKPT] Caricati pesi: {ckpt}")
    else:
        print(f"[WARN] Checkpoint non trovato: {ckpt}. Proseguo con pesi correnti.")

    train_loader, _, test_loader = get_loader(
        dataset_name=dataset,
        batch_size=getattr(SETTINGS, "BATCH_SIZE", 256),
        network_name=network
    )

    if hasattr(model, "quantize_model"):
        model.to("cpu").eval()
        newm = model.quantize_model(calib_loader=train_loader)
        if newm is not None:
            model = newm
        model.eval()
        print("[PTQ] Quantizzazione 8-bit completata (FBGEMM/CPU).")

    # cache dei batch
    test_batches: List[tuple] = list(test_loader)

    # golden by-batch
    clean_by_batch: List[torch.Tensor] = []
    with torch.inference_mode():
        for xb, _ in test_batches:
            logits = model(xb.to("cpu"))
            clean_by_batch.append(torch.argmax(logits, dim=1).cpu())

    # baseline
    all_clean = torch.cat(clean_by_batch) if clean_by_batch else torch.empty(0, dtype=torch.long)
    num_classes = int(all_clean.max().item() + 1) if all_clean.numel() else 2
    baseline_hist = np.bincount(all_clean.numpy(), minlength=num_classes)
    baseline_dist = baseline_hist / max(1, baseline_hist.sum())
    print(f"[BASELINE] pred dist = {baseline_dist.tolist()}")

    return model, test_batches, clean_by_batch, baseline_hist, baseline_dist, num_classes, dataset, network


# ---------------- Selezione batch via Hoeffding (OFF di default per esaustiva piena) ----------------

def compute_subset_idx(test_batches: List[tuple], eps: float, delta: float) -> List[int]:
    if eps <= 0 or delta <= 0:
        return list(range(len(test_batches)))
    need = math.ceil((1.0 / (2 * eps * eps)) * math.log(2.0 / delta))
    bs = None
    for xb, _ in test_batches:
        if hasattr(xb, "shape") and len(xb) > 0:
            bs = int(len(xb)); break
    if not bs:
        return list(range(len(test_batches)))
    nb = max(1, min(len(test_batches), math.ceil(need / bs)))
    idx = np.linspace(0, len(test_batches) - 1, nb, dtype=int).tolist()
    return sorted(set(idx))


# ---------------- Valutazione UNA combinazione ----------------

def _evaluate_combo(model: torch.nn.Module,
                    loader: List[tuple],
                    clean_by_batch: List[torch.Tensor],
                    injector: WeightFaultInjector,
                    combo: Tuple[Tuple[str, tuple, int], ...],
                    inj_id: int) -> float:
    global SUBSET_IDX
    if SUBSET_IDX is None:
        SUBSET_IDX = list(range(len(loader)))
    total = sum(len(clean_by_batch[b]) for b in SUBSET_IDX)
    mism = 0
    faults = [WeightFault(injection=inj_id, layer_name=ln, tensor_index=ti, bits=[bt])
              for (ln, ti, bt) in combo]
    try:
        injector.inject_faults(faults, 'bit-flip')
        with torch.inference_mode():
            for b in SUBSET_IDX:
                xb, _ = loader[b]
                y_f = torch.argmax(model(xb.to("cpu")), dim=1).cpu()
                mism += int((y_f != clean_by_batch[b]).sum().item())
    finally:
        injector.restore_golden()
    return mism / float(total) if total > 0 else 0.0


# ---------------- ESAUSTIVA su combination-space ----------------

def run_exhaustive_samepop(model,
                           test_batches: List[tuple],
                           clean_by_batch: List[torch.Tensor],
                           K: int,
                           dataset: str,
                           network: str,
                           baseline_dist,
                           target_modules: Optional[List[str]],
                           bits: Optional[List[int]],
                           save_dir: str,
                           top_k: int = 100,
                           max_combos_global: Optional[int] = None,
                           shards_total: int = 1,
                           shard_idx: int = 0):
    t0 = time.time()

    all_faults = _build_all_faults(model, target_modules=target_modules, bits=bits, as_list=True)
    F = len(all_faults)
    if F == 0:
        raise RuntimeError("Nessun sito di fault quantizzato trovato.")
    if K > F:
        print(f"[WARN] K={K} > F={F}, ridimensiono K=F.")
        K = F

    try:
        N_total = math.comb(F, K)
    except Exception:
        N_total = int(round(math.exp(math.lgamma(F + 1) - math.lgamma(K + 1) - math.lgamma(F - K + 1))))

    if isinstance(max_combos_global, str):
        try: max_combos_global = int(max_combos_global)
        except Exception: max_combos_global = None

    if isinstance(max_combos_global, int) and max_combos_global > 0:
        N_cap = min(N_total, max_combos_global)
        print(f"[EXA] F={F}  K={K}  C(F,K)≈{_sci_format_comb(F,K)} (esatto {N_total}) | CAP globale={N_cap}")
    else:
        N_cap = N_total
        print(f"[EXA] F={F}  K={K}  C(F,K)≈{_sci_format_comb(F,K)} (esatto {N_total}) | CAP globale=None")

    combos = combinations(all_faults, K)
    combos_capped = islice(combos, N_cap)

    injector = WeightFaultInjector(model)

    sum_fr = 0.0
    n_inj = 0
    top_heap = []

    approx_in_shard = max(1, N_cap // max(1, shards_total))
    desc = f"Exhaustive K={K} | shard {shard_idx+1}/{shards_total}"
    pbar = tqdm(total=approx_in_shard, desc=desc, mininterval=1.0)

    taken_in_shard = 0
    global_i = 0
    for combo in combos_capped:
        take = (global_i % max(1, shards_total)) == shard_idx
        global_i += 1
        if not take: continue

        inj_id = global_i
        fr = _evaluate_combo(model, test_batches, clean_by_batch, injector, combo, inj_id)
        sum_fr += fr; n_inj += 1; taken_in_shard += 1

        if len(top_heap) < top_k:
            heapq.heappush(top_heap, (fr, inj_id, combo))
        elif fr > top_heap[0][0]:
            heapq.heapreplace(top_heap, (fr, inj_id, combo))

        if (taken_in_shard % 100) == 0:
            pbar.n = min(taken_in_shard, approx_in_shard); pbar.refresh()

    pbar.n = min(taken_in_shard, approx_in_shard); pbar.close()

    avg_fr = (sum_fr / n_inj) if n_inj else 0.0

    # salvataggio shard
    bs = getattr(SETTINGS, "BATCH_SIZE", None) or (len(test_batches[0][0]) if len(test_batches) else 0)
    save_dir = _env_or_settings("SAVE_DIR", save_dir)
    out_dir = os.path.join(save_dir, dataset, network, f"batch_{bs}", "exhaustive_samepop")
    os.makedirs(out_dir, exist_ok=True)
    prefix = f"{dataset}_{network}_EXACTCOMB_K{K}_batch{bs}_sh{shards_total}_idx{shard_idx}"
    out_file = os.path.join(out_dir, f"{prefix}.txt")

    top_sorted = sorted(top_heap, key=lambda x: x[0], reverse=True)
    with open(out_file, "w") as f:
        f.write(f"[EXA] SAME-POP | F={F}  K={K}\n")
        f.write(f"N_total=C(F,K)={N_total}  N_cap_global={N_cap}  shards_total={shards_total}  shard_idx={shard_idx}\n")
        f.write(f"FRcrit_avg={avg_fr:.8f}  n_evaluated_in_shard={n_inj}\n\n")
        f.write(f"baseline_pred_dist={baseline_dist.tolist()}\n")
        f.write(f"Top-{min(top_heap.__len__(), len(top_sorted))} worst injections (shard {shard_idx}/{shards_total})\n")
        for rank, (fr, inj_id, combo) in enumerate(top_sorted, 1):
            desc = ", ".join(f"{ln}[{ti}] bit{bt}" for (ln, ti, bt) in combo)
            f.write(f"{rank:3d}) Inj {inj_id:9d} | FRcrit={fr:.6f} | {desc}\n")

    dt = (time.time() - t0) / 60.0
    print(f"[EXA] salvato {out_file} – {dt:.2f} min | FR={avg_fr:.6f} | n(shard)={n_inj} | N_cap={N_cap}")
    return avg_fr, n_inj, out_file, out_dir, bs


# ---------------- MERGE automatico degli shard ----------------

_pat_fr = re.compile(r"FRcrit_avg=([0-9.]+)")
_pat_n  = re.compile(r"n_evaluated_in_shard=(\d+)")
_pat_hdr= re.compile(r"_EXACTCOMB_K(\d+)_batch(\d+)_sh(\d+)_idx(\d+)\.txt$")

def _parse_one(fname: str):
    txt = open(fname, "r", encoding="utf-8").read()
    fr  = float(_pat_fr.search(txt).group(1))
    n   = int(_pat_n.search(txt).group(1))
    return fr, n

def merge_shards(out_dir: str, dataset: str, network: str, batch: int, K: int):
    import glob
    patt = os.path.join(out_dir, f"{dataset}_{network}_EXACTCOMB_K{K}_batch{batch}_sh*_idx*.txt")
    files = sorted(glob.glob(patt))
    if not files:
        print(f"[MERGE] Nessun file shard per K={K}."); return None
    lst = []
    for f in files:
        try: lst.append(_parse_one(f))
        except Exception as e:
            print(f"[MERGE] Skipping {f}: {e}")
    if not lst:
        print(f"[MERGE] Nessun dato valido per K={K}."); return None
    wsum = sum(fr*n for fr, n in lst)
    nsum = sum(n for _, n in lst)
    fr_w = wsum / nsum if nsum else 0.0
    merged = {
        "dataset": dataset, "network": network, "batch": batch, "K": K,
        "n_files": len(lst), "n_total": nsum, "FRcrit_weighted": fr_w
    }
    out_json = os.path.join(out_dir, f"{dataset}_{network}_EXACTCOMB_K{K}_batch{batch}_MERGED.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    print(f"[MERGE] K={K} -> FRcrit globale={fr_w:.8f} (n_total={nsum}) -> {out_json}")
    return merged


# ---------------- Worker per autospawn ----------------

def _worker_autospawn(K: int, shard_idx: int, shards_total: int):
    try:
        torch.set_num_threads(psutil.cpu_count(logical=False) or 1 if psutil else 1)
    except Exception:
        torch.set_num_threads(1)

    (model, test_batches, clean_by_batch,
     baseline_hist, baseline_dist, num_classes, dataset, network) = build_and_quantize_once()

    eps = float(_env_or_settings("EXA_TEST_EPS", 0))
    delta = float(_env_or_settings("EXA_TEST_DELTA", 1e-3))
    idx = compute_subset_idx(test_batches, eps, delta)
    global SUBSET_IDX; SUBSET_IDX = idx
    if eps > 0:
        totb = len(test_batches)
        print(f"[HOEFFDING] eps={eps} delta={delta} -> uso {len(idx)}/{totb} batch (~{100*len(idx)/totb:.1f}%).")

    TARGET_MODULES = _env_or_settings("EXA_TARGET_MODULES", [])
    EXA_BITS      = _env_or_settings("EXA_BITS", list(range(8)))
    SAVE_DIR      = _env_or_settings("SAVE_DIR", "results_minimal")
    TOP_K         = _env_or_settings("TOP_K", 100)
    MAX_COMBOS    = _env_or_settings("EXA_MAX_COMBOS", None)

    run_exhaustive_samepop(
        model=model,
        test_batches=test_batches,
        clean_by_batch=clean_by_batch,
        K=K,
        dataset=dataset,
        network=network,
        baseline_dist=baseline_dist,
        target_modules=TARGET_MODULES,
        bits=EXA_BITS,
        save_dir=SAVE_DIR,
        top_k=TOP_K,
        max_combos_global=MAX_COMBOS,
        shards_total=shards_total,
        shard_idx=shard_idx
    )


# ---------------- Main ----------------

if __name__ == "__main__":
    # processo principale: usa tutti i core (i worker useranno 1 thread)
    try:
        torch.set_num_threads(psutil.cpu_count(logical=True) if psutil else (os.cpu_count() or 1))
    except Exception:
        torch.set_num_threads(os.cpu_count() or 1)

    # Parametri globali
    K_list          = _env_or_settings("EXA_K_LIST", [1, 2, 3])
    TARGET_MODULES  = _env_or_settings("EXA_TARGET_MODULES", [])
    EXA_BITS        = _env_or_settings("EXA_BITS", list(range(8)))
    MAX_COMBOS      = _env_or_settings("EXA_MAX_COMBOS", None)
    TOP_K           = _env_or_settings("TOP_K", 100)
    SAVE_DIR        = _env_or_settings("SAVE_DIR", "results_minimal")
    SHARDS_TOTAL    = max(1, int(_env_or_settings("EXA_SHARDS_TOTAL", 1)))
    SHARD_IDX       = int(_env_or_settings("EXA_SHARD_IDX", 0))
    AUTOSPAWN       = bool(_env_or_settings("EXA_AUTOSPAWN", 0))
    PROCS           = int(_env_or_settings("EXA_PROCS", os.cpu_count() or 1))
    TEST_EPS        = float(_env_or_settings("EXA_TEST_EPS", 0))
    TEST_DELTA      = float(_env_or_settings("EXA_TEST_DELTA", 1e-3))

    if SHARD_IDX < 0 or SHARD_IDX >= SHARDS_TOTAL:
        raise RuntimeError(f"EXA_SHARD_IDX fuori range [0..{SHARDS_TOTAL-1}]")

    print(f"[PARAM] K_list={K_list} | TARGET_MODULES={TARGET_MODULES} | BITS={EXA_BITS} | MAX_COMBOS={MAX_COMBOS}")
    print(f"[PARAM] SHARDS_TOTAL={SHARDS_TOTAL}  SHARD_IDX={SHARD_IDX} | AUTOSPAWN={int(AUTOSPAWN)} PROCS={PROCS}")
    print(f"[PARAM] SAVE_DIR={SAVE_DIR} | TOP_K={TOP_K} | TEST_EPS={TEST_EPS} TEST_DELTA={TEST_DELTA}")

    if AUTOSPAWN:
        from multiprocessing import get_start_method, set_start_method, Process
        try:
            if get_start_method(allow_none=True) is None:
                set_start_method("spawn")
        except Exception:
            pass

        # Ogni K: spawn PROCS shard e poi MERGE
        for K in K_list:
            procs: List[Process] = []
            for idx in range(max(1, PROCS)):
                p = Process(target=_worker_autospawn, args=(K, idx, max(1, PROCS)))
                procs.append(p)
            for p in procs: p.start()
            for p in procs: p.join()

            # Dopo che tutti gli shard hanno scritto i loro file: MERGE
            # (ricostruisco out_dir/batch per cercare i file)
            (model, test_batches, _, _, _, _, dataset, network) = build_and_quantize_once()
            bs = getattr(SETTINGS, "BATCH_SIZE", None) or (len(test_batches[0][0]) if len(test_batches) else 0)
            out_dir = os.path.join(SAVE_DIR, dataset, network, f"batch_{bs}", "exhaustive_samepop")
            merge_shards(out_dir, dataset, network, bs, K)

        raise SystemExit(0)

    # --------- Modalità singolo processo (con eventuale sharding manuale) ---------
    (model, test_batches, clean_by_batch,
     baseline_hist, baseline_dist, num_classes, dataset, network) = build_and_quantize_once()

    SUBSET_IDX = compute_subset_idx(test_batches, TEST_EPS, TEST_DELTA)
    if TEST_EPS > 0:
        totb = len(test_batches)
        print(f"[HOEFFDING] eps={TEST_EPS} delta={TEST_DELTA} -> uso {len(SUBSET_IDX)}/{totb} batch (~{100*len(SUBSET_IDX)/totb:.1f}%).")

    last_out_dir, bs = None, getattr(SETTINGS, "BATCH_SIZE", None) or (len(test_batches[0][0]) if len(test_batches) else 0)
    for K in K_list:
        _, _, _, out_dir, bs = run_exhaustive_samepop(
            model=model,
            test_batches=test_batches,
            clean_by_batch=clean_by_batch,
            K=K,
            dataset=dataset,
            network=network,
            baseline_dist=baseline_dist,
            target_modules=TARGET_MODULES,
            bits=EXA_BITS,
            save_dir=SAVE_DIR,
            top_k=TOP_K,
            max_combos_global=MAX_COMBOS,
            shards_total=SHARDS_TOTAL,
            shard_idx=SHARD_IDX
        )
        last_out_dir = out_dir
        # MERGE (anche se un solo shard: ottieni comunque file *MERGED.json* omogeneo)
        merge_shards(out_dir, dataset, network, bs, K)
