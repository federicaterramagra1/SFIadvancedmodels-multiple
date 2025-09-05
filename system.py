import pandas as pd

# Carica il file
df = pd.read_csv("/home/f.terramagra/SFIadvancedmodels-multiple/results_summary/Banknote/SimpleMLP/batch_64/online_summary_N3.csv")

# Estrai la colonna 'failure_rate'
fr = df['failure_rate']

# Calcola min, max, media, percentili
print("Failure Rate - Min:", fr.min())
print("Failure Rate - Max:", fr.max())
print("Failure Rate - Mean:", fr.mean())
print("Failure Rate - Percentili 5-50-95:", fr.quantile([0.05, 0.5, 0.95]))
