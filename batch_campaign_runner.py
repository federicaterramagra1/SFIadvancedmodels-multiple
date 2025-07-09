from math import comb
from main_online_minimal import run_online_fault_injection_minimal
import pickle
import os

def compute_sample_size(N, e=0.01, t=2.58, p=0.5):
    numerator = t**2 * p * (1 - p)
    denominator = e**2 + (numerator / N)
    return int(numerator / denominator)

BATCH_SIZE = 16000
N_BATCH = 10

# Questi sono i batch random da 16k per ciascun N
BATCHED_SMALL_N = [5, 6, 7, 8, 9, 10]

# Questi sono i batch da 16k da faultlist pre-generata
LARGE_N_LIST = [50, 100, 150, 200, 250, 288]

print("=== INIZIO CAMPAGNA BATCH ===")

# ---- 1. Batch random per SMALL N ----
for n in BATCHED_SMALL_N:
    N_total = comb(288, n)
    sample_size = compute_sample_size(N_total)
    for batch_id in range(1, N_BATCH + 1):
        print(f"\n--- N = {n} | Batch {batch_id}/{N_BATCH} | Totale combinazioni = {N_total} | Sample = {sample_size}")
        run_online_fault_injection_minimal(n, sample_size, batch_idx=batch_id)

# ---- 2. Batch da lista unica per LARGE N ----
for n in LARGE_N_LIST:
    faultlist_path = f"faultlist_N{n}_160k.pkl"
    if not os.path.exists(faultlist_path):
        print(f"[ERROR] Faultlist file {faultlist_path} non trovato! Devi prima generare la lista con lo script apposito.")
        continue

    print(f"\n=== Lancio campagne batch su N={n} (10 batch da 16k combinazioni uniche) ===")
    with open(faultlist_path, "rb") as f:
        fault_list = pickle.load(f)

    if len(fault_list) < BATCH_SIZE * N_BATCH:
        print(f"[WARNING] Faultlist {faultlist_path} contiene solo {len(fault_list)} combinazioni, meno di {BATCH_SIZE*N_BATCH}.")
        continue

    for batch_id in range(N_BATCH):
        start = batch_id * BATCH_SIZE
        stop = (batch_id + 1) * BATCH_SIZE
        fault_combos = fault_list[start:stop]
        print(f"--- Batch {batch_id+1}/{N_BATCH} | N={n} | Faults: {len(fault_combos)}")
        run_online_fault_injection_minimal(n, fault_combos, batch_idx=batch_id+1)

print("\nCampagna completata")
