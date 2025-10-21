#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time

import SETTINGS

#  importa le funzioni dal tuo file principale 
from main_online_minimal import (
    build_and_quantize_once,
    _build_all_faults,
    run_statistical_srs_campaign,
)

def _timestamp():
    return time.strftime("%Y%m%d-%H%M%S")

def main():
    # === FR esaustivi già noti per Banknote ===
    # Assumo ordine: N=1 -> 0.00450, N=2 -> 0.00820, N=3 -> 0.01143
    KNOWN_FR_EXH = {1: 0.00450, 2: 0.00820, 3: 0.01143}

    # 1) Build + quantize + clean pass (una sola volta)
    model, device, test_loader, clean_by_batch, baseline_hist, baseline_dist, num_classes = build_and_quantize_once()
    bs = getattr(test_loader, "batch_size", SETTINGS.BATCH_SIZE)

    # 2) Enumerazione siti single-bit (serve per il campionamento statistico)
    all_faults = _build_all_faults(model, as_list=True)

    # 3) Run SOLO statistico con Wilson e confronto con i FR esaustivi noti
    tag = f"_FASTCHECK_{_timestamp()}"
    rows = []
    for N, fr_exh in KNOWN_FR_EXH.items():
        avg_frcrit, ci, n_used, _top, out_file = run_statistical_srs_campaign(
            model, device, test_loader, clean_by_batch, all_faults, N,
            baseline_hist, baseline_dist, num_classes,
            pilot=getattr(SETTINGS, "STAT_PILOT", 1000),
            eps=getattr(SETTINGS, "STAT_EPS", 0.005),
            conf=getattr(SETTINGS, "STAT_CONF", 0.95),
            block=1,  # controllo Wilson ad ogni iniezione post-pilot → niente overshoot grosso
            budget_cap=getattr(SETTINGS, "STAT_BUDGET_CAP", None),
            seed=getattr(SETTINGS, "SEED", 0),
            save_dir="results_minimal",
            dataset_name=SETTINGS.DATASET_NAME,
            net_name=SETTINGS.NETWORK_NAME,
            bs=bs,
            prefix_tag=tag  # evita overwrite e marca i file come "FASTCHECK"
        )
        low, high = ci
        inside = (low <= fr_exh <= high)
        print(f"N={N} | EXH={fr_exh:.6f} | STAT={avg_frcrit:.6f} CI=[{low:.6f},{high:.6f}] n={n_used} | inside? {inside}")
        rows.append((N, fr_exh, avg_frcrit, low, high, n_used, inside, out_file))

    # 4) Salva un CSV riassuntivo
    base_dir = os.path.join("results_minimal", SETTINGS.DATASET_NAME, SETTINGS.NETWORK_NAME, f"batch_{bs}")
    os.makedirs(base_dir, exist_ok=True)
    csv_path = os.path.join(base_dir, f"fast_ci_vs_exh{tag}.csv")
    with open(csv_path, "w") as f:
        f.write("N,FR_exh,FR_stat,CI_low,CI_high,n_stat,inside,file\n")
        for N, fr_exh, fr_stat, low, high, n_used, inside, out_file in rows:
            f.write(f"{N},{fr_exh:.8f},{fr_stat:.8f},{low:.8f},{high:.8f},{n_used},{inside},{out_file}\n")
    print(f"[OK] CSV salvato: {csv_path}")

if __name__ == "__main__":
    main()
