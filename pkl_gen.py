import pickle
import random
from math import comb
from itertools import combinations

N_TOTAL_FAULTS = 288
N_LIST = [50, 100, 150, 200, 250, 288]
NUM_COMBINATIONS = 160000  # 160k

for n in N_LIST:
    max_combos = comb(N_TOTAL_FAULTS, n)
    if max_combos < NUM_COMBINATIONS:
        print(f"ATTENZIONE: N={n}, esistono solo {max_combos} combinazioni possibili (<160000). Verranno salvate tutte.")
        all_combos = list(combinations(range(N_TOTAL_FAULTS), n))
        combos_list = all_combos
    elif n == N_TOTAL_FAULTS:
        # Caso N=288: solo una combinazione possibile
        print(f"ATTENZIONE: N={n}, esiste solo UNA combinazione. Verrà salvata solo quella.")
        combos_list = [tuple(range(N_TOTAL_FAULTS))]
    else:
        print(f"Generating {NUM_COMBINATIONS} unique random combinations for N={n}...")
        universe = list(range(N_TOTAL_FAULTS))
        combos_set = set()
        while len(combos_set) < NUM_COMBINATIONS:
            combo = tuple(sorted(random.sample(universe, n)))
            combos_set.add(combo)
            if len(combos_set) % 10000 == 0:
                print(f"  Progress: {len(combos_set)} / {NUM_COMBINATIONS}")
        combos_list = list(combos_set)
    filename = f"faultlist_N{n}_160k.pkl"
    with open(filename, "wb") as f:
        pickle.dump(combos_list, f)
    print(f"Saved {len(combos_list)} combinations to {filename}")

# Test di apertura (opzionale)
for n in N_LIST:
    filename = f"faultlist_N{n}_160k.pkl"
    with open(filename, "rb") as f:
        fault_list = pickle.load(f)
    print(f"N={n}: {len(fault_list)} combinazioni salvate.")
