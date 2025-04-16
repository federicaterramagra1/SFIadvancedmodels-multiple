import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_failure_rates(csv_path):
    # Carica il file CSV
    summary_df = pd.read_csv(csv_path)

    # Estrai i layer come liste da stringhe
    def parse_list(s):
        try:
            return eval(s)
        except:
            return []

    summary_df['Parsed_Layers'] = summary_df['Layers'].apply(parse_list)

    # Calcolo del failure rate medio
    mean_failure_rate = summary_df['failure_rate'].mean()

    # Analisi per layer
    layer_failure_df = summary_df.explode('Parsed_Layers')
    layer_mean_failure = layer_failure_df.groupby('Parsed_Layers')['failure_rate'].mean().sort_values(ascending=False)

    # Grafico 1: Distribuzione del failure rate
    fig1, ax1 = plt.subplots()
    sns.histplot(summary_df['failure_rate'], bins=30, kde=True, ax=ax1)
    ax1.set_title("Distribuzione del failure rate per gruppo di fault")
    ax1.set_xlabel("Failure rate (SDC-1 / totale)")
    ax1.set_ylabel("Frequenza")
    fig1.savefig("failure_rate_distribution.png")

    # Grafico 2: Failure rate medio per layer
    fig2, ax2 = plt.subplots()
    layer_mean_failure.plot(kind='bar', ax=ax2)
    ax2.set_title("Failure rate medio per layer")
    ax2.set_ylabel("Failure rate medio")
    ax2.set_xlabel("Layer")
    fig2.savefig("failure_rate_per_layer.png")

    print("\n Layer più vulnerabili (failure rate medio):")
    print(layer_mean_failure.head(3))

    # Top 10 Injection più critiche
    print("\n Top 10 gruppi di fault più critici (failure_rate più alto):")
    top50 = summary_df.sort_values(by="failure_rate", ascending=False).head(50)
    print(top50[['Injection', 'Layers', 'TensorIndices', 'Bits', 'failure_rate']])
    top50.to_csv("top50faults.csv", index=False)

    # Medie per layer
    layer_stats = layer_failure_df.groupby("Parsed_Layers")[["accuracy", "failure_rate"]].mean()
    print("\nMedia accuracy e failure_rate per layer:")
    print(layer_stats)

    # Mostra i grafici a video (opzionale)
    plt.show()

    return mean_failure_rate


# Esegui l’analisi sul file
csv_path = "/home/f.terramagra/SFIadvancedmodels-multiple/results_summary/Banknote/SimpleMLP/batch_64/SimpleMLP_summary.csv"
analyze_failure_rates(csv_path)
