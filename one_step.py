"""
One Step EP (non-iterativo) – calcolo automatico con **FPC**, senza argomenti CLI.
Esegue tutto con `python one_step.py` e salva **automaticamente** i risultati in TXT e CSV.

Output:
  results_minimal/{DATASET}/{NETWORK}/one_step_fpc/
    - one_step_{DATASET}_{NETWORK}.txt  (tab-delimited, con header)
    - one_step_{DATASET}_{NETWORK}.csv  (stesse colonne)

Parametri modificabili (in alto):
  K_LIST, E, Z, P0, SEED, EXHAUSTIVE_CAP
"""

import os
import math
import random
from itertools import product, combinations, islice
from typing import Optional, Iterable, Tuple, List

import csv
import numpy as np
import torch
from tqdm import tqdm

from faultManager.WeightFault import WeightFault
from faultManager.WeightFaultInjector import WeightFaultInjector
from utils import (
    get_loader, load_from_dict, get_network,
    load_quantized_model, save_quantized_model
)
from decimal import Decimal, getcontext
getcontext().prec = 50  # precisione alta per i calcoli con N grandi

import SETTINGS

# ======= PARAMETRI MODIFICABILI =======
K_LIST = [384, 576, 768, 960, 1104, 1408, 1728, 1984, 2048, 2208]  # modifica qui se vuoi
E = 0.005         # half-width target
Z = 1.96          # ~95%
P0 = 0.5          # pianificazione conservativa
SEED = 0
EXHAUSTIVE_CAP = 0  # se >0, calcola f_exhaustive solo se N <= cap; 0 = disabilitato
# =====================================

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

def log10_comb(n: int, k: int) -> float:
    """log10( C(n,k) ) senza materializzare C(n,k)."""
    if k < 0 or k > n:
        return float("-inf")
    k = min(k, n - k)
    if k == 0:
        return 0.0
    ln10 = math.log(10.0)
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)) / ln10

def safe_comb_int(n: int, k: int, max_log10: float = 18.0) -> Tuple[Optional[int], float]:
    """
    Ritorna (C(n,k) se 'piccolo' altrimenti None, log10(C(n,k))).
    Soglia default ≈ 1e18 per restare in zona 'comoda' coi float quando serve.
    """
    lg = log10_comb(n, k)
    if lg <= max_log10:
        return math.comb(n, k), lg
    return None, lg

# ---------- EP planning (CON FPC) ----------
def plan_n_one_step(z: float, e: float, p0: float, N_pop: Optional[int]) -> Tuple[int, int]:
    """
    Ritorna (n_inf, n_fpc) con FPC robusto:
      n_inf  = ceil(z^2 p0(1-p0) / e^2)
      n_fpc  = ceil( N * A / ( (N-1)*e^2 + A ) ), clampato a [1, N], con A = z^2 p0(1-p0)
    Se N_pop è None o talmente grande da rendere trascurabile la FPC, usa n_fpc ≈ n_inf.
    """
    A = z*z*p0*(1.0 - p0)
    n_inf = max(1, math.ceil(A / (e*e)))

    # Se non abbiamo N esatto (o è enorme), FPC ~ 1 -> n_fpc ≈ n_inf
    if N_pop is None:
        return n_inf, n_inf

    # Se N è >> n_inf (criterio conservativo), FPC irrilevante
    if N_pop >= 100 * n_inf:
        return n_inf, n_inf

    # Calcolo preciso con Decimal, evitando cast in float
    A_d  = (Decimal(str(z))**2) * Decimal(str(p0)) * (Decimal(1) - Decimal(str(p0)))
    e2_d = (Decimal(str(e))**2)
    N_d  = Decimal(N_pop)

    denom = e2_d * (N_d - 1) + A_d
    nfpc_d = (N_d * A_d) / denom
    n_fpc = int(nfpc_d.to_integral_value(rounding="ROUND_CEILING"))
    n_fpc = max(1, min(N_pop, n_fpc))
    return n_inf, n_fpc

def halfwidth_normal_fpc(p_hat: float, n: int, z: float, N_pop: Optional[int]) -> float:
    """
    Half-width normale con FPC opzionale:
      hw = z * sqrt( p(1-p)/n ) * sqrt((N-n)/(N-1))
    Se N_pop è None → niente FPC (campionamento 'infinito' / con reinserimento).
    """
    p = min(max(p_hat, 1e-12), 1.0 - 1e-12)
    base = z * math.sqrt(p * (1.0 - p) / max(1, n))
    if N_pop and N_pop > 1 and 1 <= n <= N_pop:
        fpc_num = max(0.0, (N_pop - n))
        fpc_den = max(1.0, (N_pop - 1))
        fpc = math.sqrt(fpc_num / fpc_den)
        return base * fpc
    return base

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

    ds_name = getattr(SETTINGS, "DATASET_NAME", getattr(SETTINGS, "DATASET", ""))
    net_name = getattr(SETTINGS, "NETWORK_NAME", getattr(SETTINGS, "NETWORK", ""))

    print(f"Loading network {net_name} for {ds_name} ...")

    # Prepara i loader (test sempre; train solo se serve calibrare)
    print(f"Loading {ds_name} dataset...")
    train_loader, _, test_loader = get_loader(
        dataset_name=ds_name,
        batch_size=getattr(SETTINGS, "BATCH_SIZE", 64),
        network_name=net_name
    )

    # 1) Prova a caricare il modello quantizzato già salvato
    qmodel, qpath = load_quantized_model(ds_name, net_name, device="cpu", engine="fbgemm")
    if qmodel is not None:
        model = qmodel
        print(f"[PTQ] Quantized model caricato: {qpath}")
    else:
        # 2) Fallback: costruisci float, carica ckpt e fai PTQ UNA volta, poi salva
        model = get_network(net_name, device, ds_name)
        model.to(device).eval()
        ckpt = f"./trained_models/{ds_name}_{net_name}_trained.pth"
        if os.path.exists(ckpt):
            load_from_dict(model, device, ckpt)
            print("modello float caricato")
        else:
            print(f"checkpoint non trovato: {ckpt} (proseguo senza)")

        if hasattr(model, "quantize_model"):
            model.quantize_model(calib_loader=train_loader)
            model.eval()
            qsave = save_quantized_model(model, ds_name, net_name, engine="fbgemm")
            print(f"[PTQ] quantizzazione completata e salvata in: {qsave}")
        else:
            print("modello non quantizzabile (salto quantizzazione)")

    # Clean predictions (post-PTQ) per batch
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

    # Calcolo 'sicuro' di N e stringa scientifica
    N_pop_exact, _ = safe_comb_int(M, K, max_log10=18.0)  # ~ fino a 1e18
    N_pop_str = sci_comb(M, K)

    # Pianificazione (usa N esatto se disponibile, altrimenti None => n_fpc≈n_inf)
    n_inf, n_fpc = plan_n_one_step(z=z, e=e, p0=p0, N_pop=N_pop_exact)
    n_to_draw = n_fpc

    # (N-1)/N formattato a 9 decimali senza overflow
    if N_pop_exact is not None and N_pop_exact > 0:
        ratio_val = 1.0 - (1.0 / float(N_pop_exact))
        ratio = f"{ratio_val:.9f}"
    else:
        # per N astronomici, a 9 decimali è indistinguibile da 1.000000000
        ratio = "1.000000000"

    injector = WeightFaultInjector(model)
    total_samples = sum(len(t) for t in clean_by_batch)
    sum_fr = 0.0

    it = sample_unique_combos(all_faults, K, n_to_draw, seed=seed)
    it = tqdm(it, total=n_to_draw, desc=f"[OneStep] K={K}", disable=not show_progress)
    for inj_id, combo in enumerate(it, 1):
        sum_fr += evaluate_combo(model, device, test_loader, clean_by_batch, injector,
                                 combo, inj_id, total_samples)

    p_hat = sum_fr / n_to_draw if n_to_draw > 0 else 0.0
    err = halfwidth_normal_fpc(p_hat, n_to_draw, z, N_pop_exact)

    return dict(
        K=K,
        N_pop_exact=N_pop_exact,
        N_pop_str=N_pop_str,
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
    if exhaustive_cap <= 0:
        return None
    N_exact, _ = safe_comb_int(M, K, max_log10=18.0)
    if N_exact is None or N_exact > exhaustive_cap:
        return None
    injector = WeightFaultInjector(model)
    total_samples = sum(len(t) for t in clean_by_batch)
    sum_fr = 0.0
    for inj_id, combo in enumerate(tqdm(combinations(all_faults, K), total=N_exact, desc=f"[EXA] K={K}"), 1):
        sum_fr += evaluate_combo(model, device, test_loader, clean_by_batch, injector,
                                 combo, inj_id, total_samples)
    return sum_fr / N_exact if N_exact > 0 else 0.0

# ---------- utility: output su file ----------
def _prepare_output_handles(ds_name: str, net_name: str):
    out_dir = os.path.join("results_minimal", ds_name, net_name, "one_step_fpc")
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, f"one_step_{ds_name}_{net_name}.txt")
    csv_path = os.path.join(out_dir, f"one_step_{ds_name}_{net_name}.csv")
    txt_fh = open(txt_path, "w")
    csv_fh = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_fh)
    return txt_path, csv_path, txt_fh, csv_fh, csv_writer

def _write_header(txt_fh, csv_writer):
    header = ["K","N","ratio_(N-1)/N","f_exhaustive","n_inf","n_fpc","f_rate_EP","err_EP","n_used"]
    print("K\tN\t(N-1)/N\tf_exhaustive\t n\t n (FPC)\t f-rate (EP)\t errore eps\t n_used")
    txt_fh.write("\t".join(header) + "\n")
    csv_writer.writerow(header)

def _write_row(txt_fh, csv_writer, row_values: List[str]):
    print("\t".join(row_values))
    txt_fh.write("\t".join(row_values) + "\n")
    csv_writer.writerow(row_values)

# ---------- main ----------
def main():
    # build + clean (riusa il modello quantizzato se esiste)
    model, device, test_loader, clean_by_batch, _, _, _ = build_and_quantize_once()

    # nomi DS/NET per output
    ds_name = getattr(SETTINGS, "DATASET_NAME", getattr(SETTINGS, "DATASET", "DS"))
    net_name = getattr(SETTINGS, "NETWORK_NAME", getattr(SETTINGS, "NETWORK", "NET"))

    # output files
    txt_path, csv_path, txt_fh, csv_fh, csv_writer = _prepare_output_handles(ds_name, net_name)

    try:
        # siti di fault (M)
        all_faults = build_fault_sites(model, as_list=True)
        M = len(all_faults)
        print(f"[INFO] Total Faults (M) = {M}")
        print(f"[INFO] Salvo in:\n  TXT: {txt_path}\n  CSV: {csv_path}")

        # header
        _write_header(txt_fh, csv_writer)

        for K in K_LIST:
            if K < 1 or K > M:
                row = [str(K), "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"]
                _write_row(txt_fh, csv_writer, row)
                continue

            out = run_one_step_ep_for_K(
                model, device, test_loader, clean_by_batch,
                all_faults, K,
                z=Z, e=E, p0=P0,
                seed=SEED, show_progress=True
            )
            if out is None:
                row = [str(K), "NA", "NA", "NA", "NA", "NA", "NA", "NA", "NA"]
                _write_row(txt_fh, csv_writer, row)
                continue

            f_ex = maybe_exhaustive_f(
                model, device, test_loader, clean_by_batch,
                all_faults, K, exhaustive_cap=EXHAUSTIVE_CAP
            )
            f_ex_str = f"{f_ex:.8f}" if f_ex is not None else "NA"

            row = [
                str(out['K']),
                out['N_pop_str'],
                out['ratio'],                 # già formattato a 9 decimali
                f_ex_str,
                str(out['n_inf']),
                str(out['n_fpc']),
                f"{out['f_rate_ep']:.8f}",
                f"{out['err_ep']:.8f}",
                str(out['n_used'])
            ]
            _write_row(txt_fh, csv_writer, row)
    finally:
        try:
            txt_fh.close()
        except Exception:
            pass
        try:
            csv_fh.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
