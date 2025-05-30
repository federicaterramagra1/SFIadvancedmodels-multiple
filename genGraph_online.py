import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import SETTINGS
import os


def analyze_failure_rates():
    N = SETTINGS.NUM_FAULTS_TO_INJECT
    prefix = f"N{N}_"

    csv_path = os.path.join(
        SETTINGS.FI_ANALYSIS_PATH,
        f"online_summary_N{N}.csv"
    )
    fault_list_path = os.path.join(
        SETTINGS.FAULT_LIST_PATH,
        f"SimpleMLP_42_fault_list_N{N}.csv"
    )

    summary_df = pd.read_csv(csv_path)
    fault_df = pd.read_csv(fault_list_path)

    mean_failure_rate = summary_df['failure_rate'].mean()

    with open(f"{prefix}analysis_summary.txt", "w") as f:
        f.write(f"Failure rate medio complessivo: {mean_failure_rate:.4f}\n\n")

        # Distribuzione failure rate
        fig1, ax1 = plt.subplots()
        sns.histplot(summary_df['failure_rate'], bins=30, kde=True, ax=ax1)
        ax1.set_title("Distribuzione del failure rate per injection")
        ax1.set_xlabel("Failure rate")
        ax1.set_ylabel("Frequenza")
        fig1.savefig(f"{prefix}failure_rate_distribution.png")

        # Grafico accuracy vs failure_rate
        fig2, ax2 = plt.subplots()
        sns.scatterplot(x='accuracy', y='failure_rate', data=summary_df, ax=ax2)
        ax2.set_title("Accuracy vs Failure Rate per injection")
        ax2.set_xlabel("Accuracy")
        ax2.set_ylabel("Failure rate")
        fig2.savefig(f"{prefix}accuracy_vs_failure_rate.png")

        # Parsing layers and calculating vulnerability
        fault_df_grouped = fault_df.groupby("Injection").agg({"Layer": lambda x: list(x)})
        merged_df = summary_df.merge(fault_df_grouped, on="Injection")

        exploded_df = merged_df.explode("Layer")
        layer_mean_failure = exploded_df.groupby("Layer")['failure_rate'].mean().sort_values(ascending=False)

        f.write("Layer più vulnerabili (failure rate medio):\n")
        f.write(layer_mean_failure.to_string())
        f.write("\n\n")

        # Top 10 injection più critiche con dettagli
        detailed_df = fault_df.groupby('Injection').agg({
            'Layer': lambda x: list(x),
            'TensorIndex': lambda x: list(x),
            'Bit': lambda x: list(x)
        }).reset_index()

        top10_detailed = summary_df.merge(detailed_df, on="Injection").sort_values(by="failure_rate", ascending=False).head(10)
        f.write("Top 10 gruppi di fault più critici (failure_rate più alto):\n")
        f.write(top10_detailed[['Injection', 'Layer', 'TensorIndex', 'Bit', 'failure_rate']].to_string(index=False))
        f.write("\n\n")

        # Top 100 injection più critiche con dettagli
        top100_detailed = summary_df.merge(detailed_df, on="Injection").sort_values(by="failure_rate", ascending=False).head(100)
        top100_detailed.to_csv(f"{prefix}top100_online_faults.csv", index=False)
        f.write("Top 100 gruppi di fault più critici (failure_rate più alto):\n")
        f.write(top100_detailed[['Injection', 'Layer', 'TensorIndex', 'Bit', 'failure_rate']].to_string(index=False))
        f.write("\n\n")

        # Media accuracy e failure_rate per layer
        layer_stats = exploded_df.groupby("Layer")[["accuracy", "failure_rate"]].mean()
        layer_stats.to_csv(f"{prefix}layer_stats_summary.csv")

        f.write("Media accuracy e failure_rate per layer:\n")
        f.write(layer_stats.to_string())
        f.write("\n")

        # Mostra i grafici
        plt.show()


# Esecuzione
analyze_failure_rates()
