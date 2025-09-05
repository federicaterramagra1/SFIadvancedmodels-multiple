import pandas as pd
import numpy as np
from math import comb

# Carica il file CSV corretto già formattato
csv_path = "/home/f.terramagra/SFIadvancedmodels-multiple/failure_rate_estimates.csv"
df = pd.read_csv(csv_path)

# Parametri noti
N_tot = 288  # numero totale di bit faultabili
p_bit = 1.48e-17  # probabilità di guasto per singolo bit

# Prepara lista di risultati
results = []

# Calcola p_i con distribuzione binomiale e g = sum(f_hat_i * p_i)
for _, row in df.iterrows():
    N = row['N']
    f_hat = row['FR_AVG']

    # Ignora righe senza media (assenza di dati)
    if pd.isna(f_hat) or N > N_tot:
        continue

    # Calcolo della probabilità binomiale p_i
    try:
        p_i = comb(N_tot, int(N)) * (p_bit ** int(N)) * ((1 - p_bit) ** (N_tot - int(N)))
    except OverflowError:
        p_i = 0.0

    g_i = f_hat * p_i

    results.append({
        'N': N,
        'f_hat': f_hat,
        'p_i': p_i,
        'f_hat * p_i': g_i
    })

# Crea DataFrame dei risultati
g_df = pd.DataFrame(results)
g_df['f_hat * p_i (exp)'] = g_df['f_hat * p_i'].apply(lambda x: f"{x:.2e}")
g_df['p_i (exp)'] = g_df['p_i'].apply(lambda x: f"{x:.2e}")

# Calcola g stimato totale
g_estimate = g_df['f_hat * p_i'].sum()

import ace_tools as tools; tools.display_dataframe_to_user(name="Stima statistica g", dataframe=g_df)

g_estimate
