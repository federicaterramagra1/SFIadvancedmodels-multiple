# weight_stats_run.py
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch

# ============================== UTIL PESI & PLOT ==============================
def _get_weight_tensor(mod):
    """
    Supporta moduli float e quantizzati:
    - mod.weight() per i quantized.* (metodo)
    - mod.weight per i moduli classici (attributo)
    Ritorna (tensor, is_quantized)
    """
    if hasattr(mod, "weight"):
        w = None
        try:
            # torch.nn.quantized.* espongono weight() come metodo
            w = mod.weight()
        except TypeError:
            # moduli float classici hanno weight come attributo Tensor
            w = mod.weight
        if isinstance(w, torch.Tensor):
            return w, w.is_quantized
    return None, False


def report_weight_distributions(model: torch.nn.Module,
                                model_name: str,
                                out_dir: str,
                                bins: int = 80,
                                save_plots: bool = True):
    """
    Crea:
      - {out_dir}/weights_stats.csv  (statistiche per layer)
      - {out_dir}/plots/*.png        (istogrammi per layer + globali)
    Stampa a console le std per layer.

    Se un layer è quantizzato, salva sia l'istogramma dei float dequantizzati
    sia quello degli int8 (int_repr).
    """
    import matplotlib
    matplotlib.use("Agg")  # no-GUI
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    stats_rows = []
    all_float_weights = []

    # per layer
    for name, mod in model.named_modules():
        w, is_q = _get_weight_tensor(mod)
        if w is None:
            continue

        if is_q:  # quantizzato: prendo sia int_repr sia i float dequantizzati
            try:
                w_float = w.dequantize().detach().cpu().numpy().ravel()
            except Exception:
                # fallback prudenziale
                w_float = torch.dequantize(w).detach().cpu().numpy().ravel()
            try:
                w_int = w.int_repr().cpu().numpy().ravel()
            except Exception:
                w_int = None
        else:
            w_float = w.detach().cpu().numpy().ravel()
            w_int = None

        if w_float.size == 0:
            continue

        all_float_weights.append(w_float)

        # stats layer
        row = {
            "layer": name if name else "(root)",
            "numel": int(w_float.size),
            "mean": float(np.mean(w_float)),
            "std":  float(np.std(w_float)),
            "min":  float(np.min(w_float)),
            "max":  float(np.max(w_float)),
            "is_quantized": bool(is_q),
        }
        stats_rows.append(row)

        # grafici layer
        if save_plots:
            # float
            plt.figure()
            plt.hist(w_float, bins=bins)
            plt.title(f"{model_name} :: {name} (float)  std={row['std']:.4g}")
            plt.xlabel("weight value")
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"{name.replace('.', '_')}_float_hist.png"), dpi=140)
            plt.close()

            # int8 (se quantizzato)
            if w_int is not None:
                plt.figure()
                plt.hist(w_int, bins=256, range=(-128, 127))
                plt.title(f"{model_name} :: {name} (int8 int_repr)")
                plt.xlabel("int8 value")
                plt.ylabel("count")
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, f"{name.replace('.', '_')}_int8_hist.png"), dpi=140)
                plt.close()

    # CSV complessivo
    df = pd.DataFrame(stats_rows).sort_values(by="layer")
    df.to_csv(os.path.join(out_dir, "weights_stats.csv"), index=False)

    # istogramma globale (float, tutti i layer)
    if save_plots and len(all_float_weights) > 0:
        all_f = np.concatenate(all_float_weights, axis=0)
        plt.figure()
        plt.hist(all_f, bins=bins)
        plt.title(f"{model_name} :: ALL LAYERS (float)  std={np.std(all_f):.4g}")
        plt.xlabel("weight value")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"ALL_layers_float_hist.png"), dpi=160)
        plt.close()

    # stampa veloce a console
    print(f"\n[WeightStats] {model_name}")
    for r in df.itertuples(index=False):
        print(f"  - {r.layer:30s} std={r.std:.6f}  numel={r.numel}  q={r.is_quantized}")
    if len(all_float_weights) > 0:
        print(f"  > GLOBAL float std = {np.std(np.concatenate(all_float_weights)):.6f}")
    print(f"  CSV: {os.path.join(out_dir, 'weights_stats.csv')}")
    print(f"  Plots: {plots_dir}")


# ============================== LOAD MODEL ==============================
def _smart_load_state_dict(model: torch.nn.Module, ckpt_path: str, strict: bool = False):
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"[WARN] checkpoint non trovato: {ckpt_path}")
        return model
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        sd = state["state_dict"]
    elif isinstance(state, dict):
        sd = state
    else:
        raise RuntimeError(f"Formato checkpoint non riconosciuto: {type(state)}")

    # rimuove eventuale prefisso 'module.'
    clean_sd = {}
    for k, v in sd.items():
        k2 = k.replace("module.", "")
        clean_sd[k2] = v
    missing, unexpected = model.load_state_dict(clean_sd, strict=strict)
    if missing:
        print(f"[INFO] missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[INFO] unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    print(f"[OK] pesi caricati da {ckpt_path}")
    return model


def build_model(dataset: str, network: str):
    """
    Istanzia il modello in base a (dataset, network).
    È pensato per i tre casi richiesti; estendilo se servono altre reti.
    """
    if dataset == "Wine" and network == "WineMLP":
        from dlModels.Wine.mlp import WineMLP
        return WineMLP()
    if dataset == "DryBean" and network in ("BeanMLP", "DryBeanMLP", "BeanMLP"):
        from dlModels.DryBean.mlp import BeanMLP
        return BeanMLP()
    if dataset == "BreastCancer" and network in ("SimpleMLP", "BiggerMLP"):
        if network == "SimpleMLP":
            from dlModels.BreastCancer.mlp import SimpleMLP
            return SimpleMLP()
        else:
            from dlModels.BreastCancer.bigger_mlp import BiggerMLP
            return BiggerMLP()
    if dataset == "Banknote" and network == "SimpleMLP":
        from dlModels.Banknote.mlp import SimpleMLP
        return SimpleMLP()


    raise ValueError(f"Modello non gestito: dataset={dataset} network={network}")


def maybe_quantize_inplace(model: torch.nn.Module, dataset: str, batch_size: int = 64):
    """
    Prova a quantizzare col metodo interno del tuo modello, usando un piccolo
    loader di calibrazione (val o train ridotto). Se qualcosa manca, salta.
    """
    if not hasattr(model, "quantize_model"):
        print("[PTQ] quantize_model non trovato, salto.")
        return model

    # Calibrazione leggera in base al dataset
    calib_loader = None
    try:
        if dataset == "Wine":
            from sklearn.datasets import load_wine
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from torch.utils.data import DataLoader, TensorDataset
            data = load_wine()
            X, y = data.data, data.target
            X = StandardScaler().fit_transform(X)
            Xc, _, yc, _ = train_test_split(X, y, test_size=0.9, random_state=42, stratify=y)
            Xc = torch.tensor(Xc, dtype=torch.float32); yc = torch.tensor(yc, dtype=torch.long)
            calib_loader = DataLoader(TensorDataset(Xc, yc), batch_size=batch_size, shuffle=False)

        elif dataset == "BreastCancer":
            from sklearn.datasets import load_breast_cancer
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
            from torch.utils.data import DataLoader, TensorDataset
            data = load_breast_cancer()
            X, y = data.data, data.target
            X = StandardScaler().fit_transform(X)
            Xc, _, yc, _ = train_test_split(X, y, test_size=0.9, random_state=42, stratify=y)
            Xc = torch.tensor(Xc, dtype=torch.float32); yc = torch.tensor(yc, dtype=torch.long)
            calib_loader = DataLoader(TensorDataset(Xc, yc), batch_size=batch_size, shuffle=False)

        elif dataset == "DryBean":
            # Se hai l'ARFF nel repo: dlModels/DryBean/Dry_Bean_Dataset.arff
            from scipy.io import arff
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            from sklearn.model_selection import train_test_split
            from torch.utils.data import DataLoader, TensorDataset
            path = os.path.join("dlModels", "DryBean", "Dry_Bean_Dataset.arff")
            if os.path.exists(path):
                data, meta = arff.loadarff(path)
                import pandas as pd
                df = pd.DataFrame(data)
                if isinstance(df["Class"][0], bytes):
                    df["Class"] = df["Class"].str.decode("utf-8")
                X = df.drop("Class", axis=1).astype("float32").values
                y = LabelEncoder().fit_transform(df["Class"])
                X = StandardScaler().fit_transform(X)
                Xc, _, yc, _ = train_test_split(X, y, test_size=0.98, random_state=42, stratify=y)
                Xc = torch.tensor(Xc, dtype=torch.float32); yc = torch.tensor(yc, dtype=torch.long)
                calib_loader = DataLoader(TensorDataset(Xc, yc), batch_size=batch_size, shuffle=False)
    except Exception as e:
        print(f"[PTQ] calibrazione leggera non disponibile: {e}")

    try:
        model.eval().cpu()
        maybe_new = model.quantize_model(calib_loader=calib_loader)
        if maybe_new is not None:
            model = maybe_new
        setattr(model, "_quantized_done", True)
        print("[PTQ] Modello quantizzato in-place.")
    except Exception as e:
        print(f"[PTQ] Quantizzazione fallita/assente: {e}")

    return model


# ============================== MAIN ==============================
def main():
    parser = argparse.ArgumentParser(description="Report distribuzione pesi + std per più reti")
    parser.add_argument("--config", type=str, default="",
                        help="JSON con la lista di target; se assente usa TARGETS nel file")
    parser.add_argument("--quantize", action="store_true",
                        help="Tenta PTQ (se il modello espone quantize_model)")
    parser.add_argument("--bins", type=int, default=80, help="Bins degli istogrammi float")
    parser.add_argument("--no-plots", action="store_true", help="Non salvare i grafici")
    args = parser.parse_args()

    # percorsi base assoluti
    MODEL_DIR = "/home/f.terramagra/SFIadvancedmodels-multiple/trained_models"
    OUT_BASE  = "/home/f.terramagra/SFIadvancedmodels-multiple/results_summary"

    TARGETS = [
        {"dataset": "Wine", "network": "WineMLP",
        "ckpt": f"{MODEL_DIR}/Wine_WineMLP_trained.pth",
        "out_dir": f"{OUT_BASE}/Wine/WineMLP/weights"},

        {"dataset": "DryBean", "network": "BeanMLP",
        "ckpt": f"{MODEL_DIR}/DryBean_BeanMLP_trained.pth",
        "out_dir": f"{OUT_BASE}/DryBean/BeanMLP/weights"},

        {"dataset": "Banknote", "network": "SimpleMLP",
        "ckpt": f"{MODEL_DIR}/Banknote_SimpleMLP_trained.pth",
        "out_dir": f"{OUT_BASE}/Banknote/SimpleMLP/weights"},
    ]


    if args.config:
        with open(args.config, "r") as f:
            TARGETS = json.load(f)

    if not TARGETS:
        print("⚠️  Nessun target definito. Modifica TARGETS nel file o passa --config <file.json>.")
        sys.exit(1)

    for t in TARGETS:
        dataset = t["dataset"]
        network = t["network"]
        ckpt    = t.get("ckpt", "")
        out_dir = t.get("out_dir", f"results_summary/{dataset}/{network}/weights")

        print(f"\n=== [{dataset} / {network}] ===")
        model = build_model(dataset, network)
        model = _smart_load_state_dict(model, ckpt)
        if args.quantize:
            model = maybe_quantize_inplace(model, dataset)

        report_weight_distributions(
            model=model,
            model_name=f"{dataset}-{network}",
            out_dir=out_dir,
            bins=args.bins,
            save_plots=not args.no_plots
        )


if __name__ == "__main__":
    main()
