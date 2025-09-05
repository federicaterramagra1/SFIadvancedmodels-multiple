import numpy as np
import pandas as pd
from math import comb

# Parametri noti
N = 536_870_912  # numero di bit nella memoria
p = 1.48e-17     # probabilità di bit flip singolo
K = 20           # numero massimo di fault multipli da considerare

# Funzione per la probabilità binomiale di avere esattamente k bit flip
def binomial_prob(k, N, p):
    return comb(N, k) * (p**k) * ((1 - p)**(N - k))

# Stima worst-case: ogni combinazione k fault causa errore => f_hat_k = 1
worst_case_FR = []
for k in range(1, K + 1):
    pi_k = binomial_prob(k, N, p)
    worst_case_FR.append((k, pi_k, 1.0, pi_k))  # (k, prob, f_hat_k, contribution)

# Modello ipotetico con severità: f_hat_k = 1 - (1 - alpha)^k
alpha = 0.05
severity_model_FR = []
for k in range(1, K + 1):
    pi_k = binomial_prob(k, N, p)
    f_hat_k = 1 - (1 - alpha)**k
    contribution = f_hat_k * pi_k
    severity_model_FR.append((k, pi_k, f_hat_k, contribution))

# Organizza i risultati
df_worst = pd.DataFrame(worst_case_FR, columns=["k", "P_k", "f_hat_k", "contribution"])
df_severity = pd.DataFrame(severity_model_FR, columns=["k", "P_k", "f_hat_k", "contribution"])

# Calcola g_bar
g_bar_worst = df_worst["contribution"].sum()
g_bar_severity = df_severity["contribution"].sum()

df_severity.to_csv("stima_fr_teorico.csv", index=False)
print("Risultato salvato in stima_fr_teorico.csv")

g_bar_worst, g_bar_severity
