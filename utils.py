import os
import math
import csv
import random
from typing import Union, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn import Sequential, Module
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import SETTINGS

# torchvision (per alcuni dataset CNN e utilità)
from torchvision import transforms
from torchvision.datasets import GTSRB, CIFAR10, CIFAR100, MNIST
from torchvision.models import resnet
from torchvision.models.densenet import _DenseBlock, _Transition
from torchvision.models.efficientnet import Conv2dNormActivation
from torchvision.transforms.v2 import (
    ToTensor, Resize, Compose, ColorJitter, RandomRotation, AugMix,
    GaussianBlur, RandomEqualize, RandomHorizontalFlip, RandomVerticalFlip
)

# sklearn / arff
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.io import arff

# FI manager (solo tipi)
from faultManager.FaultListManager import FLManager
from faultManager.NeuronFault import NeuronFault
from faultManager.WeightFault import WeightFault


# ============================== Error types ==============================

class UnknownNetworkException(Exception):
    pass


# ============================== Inference utils ==============================

@torch.no_grad()
def clean_inference(network, loader, device, network_name):
    """
    Valutazione pulita su 'loader'. Stampa wrong/accuracy in modo robusto.
    """
    network.eval()
    total = 0
    wrong = 0

    pbar = tqdm(loader, colour='green', desc='Clean Run')
    for _, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        logits = network(x)
        pred = torch.argmax(logits, dim=1)
        total += y.numel()
        wrong += (pred != y).sum().item()

    acc = (1 - wrong / max(1, total)) * 100.0
    print(f"device: {device}")
    print(f"network: {network_name}")
    print(f"Clean wrong predictions: {wrong}/{total}")
    print(f"Clean accuracy: {acc:.2f}%")


@torch.no_grad()
def faulty_inference(network, loader, device, network_name, faults_injected=False):
    """
    Valutazione del modello 'faulty'.
    """
    network.eval()
    total = 0
    wrong = 0

    pbar = tqdm(loader, colour='red', desc='Faulty Run')
    for _, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        logits = network(x)
        pred = torch.argmax(logits, dim=1)
        total += y.numel()
        wrong += (pred != y).sum().item()

    acc = (1 - wrong / max(1, total)) * 100.0
    print(f"\nFaulty inference results on device: {device}")
    print(f"Model: {network_name}")
    print(f"Wrong predictions: {wrong}")
    print(f"Faulty model accuracy: {acc:.2f}%")


# ============================== Metrics ==============================

def _confusion_matrix(num_classes, y_true, y_pred):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


@torch.no_grad()
def _evaluate(model, data_loader, device='cpu', num_classes=None):
    model.eval()
    ys, ps = [], []
    for x, y in data_loader:
        x = x.to(device)
        logits = model(x)
        ys.append(y.view(-1).cpu().numpy())
        ps.append(torch.argmax(logits, 1).cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    if num_classes is None:
        num_classes = int(y_true.max()) + 1
    cm = _confusion_matrix(num_classes, y_true, y_pred)
    acc = (y_true == y_pred).mean() * 100.0
    # macro-F1
    f1s = []
    for k in range(num_classes):
        TP = cm[k, k]
        FP = cm[:, k].sum() - TP
        FN = cm[k, :].sum() - TP
        denom = 2 * TP + FP + FN
        f1s.append(0.0 if denom == 0 else 2 * TP / denom)
    macro_f1 = float(np.mean(f1s)) * 100.0
    return acc, macro_f1, cm


def _class_weights_from_loader(train_loader, num_classes):
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, y in train_loader:
        y = y.view(-1).cpu().numpy()
        for c in np.unique(y):
            counts[c] += (y == c).sum()
    counts = np.clip(counts, 1, None)
    inv = 1.0 / counts
    return torch.tensor(inv / inv.mean(), dtype=torch.float32)


# ============================== Training ==============================

def train_model(model, train_loader, val_loader=None, num_epochs=50, lr=0.001, device='cpu', save_path=None):
    """
    Training semplice (Adam) con opzionale best-checkpoint su validation accuracy.
    """
    print(f"\n Inizio training su {device} per {num_epochs} epoche...")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        tot_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            bs = labels.size(0)
            tot_loss += loss.item() * bs
            correct += (outputs.argmax(1) == labels).sum().item()
            total += bs

        acc = 100.0 * correct / max(1, total)
        print(f" Epoch {epoch+1}/{num_epochs} | loss {tot_loss/max(1,total):.4f} | train_acc {acc:.2f}%")

        if val_loader:
            model.eval()
            val_acc, _, _ = _evaluate(model, val_loader, device=device)
            print(f"  Validation Accuracy: {val_acc:.2f}%")

            if save_path and val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                torch.save(best_state, save_path)
                print(f"  [Save] best checkpoint → {save_path}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print("[Train-basic] Best weights ripristinati.")
    print("[Train-basic] done.")
    return model


def train_model_complete(
    model,
    train_loader,
    val_loader=None,
    num_epochs=150,
    lr=1e-3,
    weight_decay=1e-4,
    label_smoothing=0.05,
    warmup_epochs=5,
    early_stop_patience=20,
    grad_clip=1.0,
    device='cpu',
    save_path=None,
    seed=123,
    do_ptq=False,                # <- di default NON quantizziamo qui (uniformità col main)
    calib_loader=None,
    USE_CLASS_WEIGHTS=True,
    REFIT_TRAINVAL=False,
    REFIT_EPOCHS=0,
    REFIT_LR=5e-4,
):
    torch.manual_seed(seed); np.random.seed(seed)
    model = model.to(device)
    print(f"\n[Train-adv] device={device} epochs={num_epochs} lr={lr} wd={weight_decay}")

    # num classi
    for _, ytmp in train_loader:
        num_classes = int(ytmp.max().item()) + 1
        break

    class_w = _class_weights_from_loader(train_loader, num_classes).to(device) if USE_CLASS_WEIGHTS else None
    criterion = torch.nn.CrossEntropyLoss(weight=class_w, label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        prog = (epoch - warmup_epochs) / max(1, (num_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_score = -1.0   # macro-F1
    best_state = None
    patience = 0

    for epoch in range(num_epochs):
        model.train()
        tot_loss = 0.0
        tot = 0
        correct = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            bs = y.size(0)
            tot_loss += loss.item() * bs
            correct += (logits.argmax(1) == y).sum().item()
            tot += bs

        scheduler.step()
        train_loss = tot_loss / max(1, tot)
        train_acc = 100.0 * correct / max(1, tot)

        if val_loader is not None:
            val_acc, val_f1, _ = _evaluate(model, val_loader, device=device, num_classes=num_classes)
            score = val_f1
            improved = score > best_score
            if improved:
                best_score = score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
                if save_path:
                    torch.save(best_state, save_path)
            else:
                patience += 1

            print(f"Epoch {epoch+1:03d}/{num_epochs} | loss {train_loss:.4f} | "
                  f"train_acc {train_acc:.2f}% | val_acc {val_acc:.2f}% | "
                  f"val_macroF1 {val_f1:.2f}% | lr {scheduler.get_last_lr()[0]:.6f} "
                  f"{'[*] best' if improved else ''}")

            if patience >= early_stop_patience:
                print(f"[EarlyStop] pazienza={early_stop_patience} a epoch {epoch+1}.")
                break
        else:
            print(f"Epoch {epoch+1:03d}/{num_epochs} | loss {train_loss:.4f} | "
                  f"train_acc {train_acc:.2f}% | lr {scheduler.get_last_lr()[0]:.6f}")

    if best_state is not None:
        model.load_state_dict(best_state)
        print("[Train-adv] Best weights ripristinati (macro-F1 val).")
    else:
        print("[Train-adv] Nessun best salvato (no val).")

    if val_loader is not None:
        val_acc, val_f1, _ = _evaluate(model, val_loader, device=device, num_classes=num_classes)
        print(f"[Float Best] val_acc={val_acc:.2f}%  val_macroF1={val_f1:.2f}%")

    # PTQ opzionale: gestiamo sia metodi in-place sia che ritornano un modello
    if do_ptq and hasattr(model, "quantize_model"):
        model.eval()
        calib = calib_loader if calib_loader is not None else (val_loader if val_loader is not None else train_loader)
        try:
            model.to('cpu')
            maybe_new = model.quantize_model(calib_loader=calib)
            if maybe_new is not None:
                model = maybe_new  # dynamic path
            setattr(model, "_quantized_done", True)
            print("[PTQ] Modello quantizzato (int8).")
            if val_loader is not None:
                q_acc, q_f1, _ = _evaluate(model, val_loader, device='cpu', num_classes=num_classes)
                print(f"[Quantized] val_acc={q_acc:.2f}%  val_macroF1={q_f1:.2f}%")
        except Exception as e:
            print(f"[PTQ] Quantizzazione fallita: {e}")

    print("[Train-adv] done.")
    return model


# ============================== Network loader ==============================

def get_network(network_name: str,
                device: torch.device,
                dataset_name: str,
                root: str = '.') -> torch.nn.Module:
    """
    Istanzia la rete corretta per il dataset. NIENTE QuantWrapper qui.
    Ogni MLP contiene già QuantStub/DeQuantStub e (se presente) quantize_model().
    """
    if dataset_name == 'BreastCancer':
        print(f'Loading network {network_name} for BreastCancer ...')
        if network_name == 'SimpleMLP':
            from dlModels.BreastCancer.mlp import SimpleMLP as _BCSimple
            network = _BCSimple()
        elif network_name == 'BiggerMLP':
            from dlModels.BreastCancer.bigger_mlp import BiggerMLP
            network = BiggerMLP()
        else:
            raise ValueError(f"Unknown network '{network_name}' for dataset '{dataset_name}'")

    elif dataset_name == 'Letter':
        print(f'Loading network {network_name} for Letter ...')
        if network_name == 'LetterMLP':
            from dlModels.Letter.mlp import LetterMLP
            network = LetterMLP()
        else:
            raise ValueError(f"Unknown network '{network_name}' for dataset 'Letter'")

    elif dataset_name == 'Banknote':
        print(f'Loading network {network_name} for Banknote ...')
        if network_name == 'SimpleMLP':
            from dlModels.Banknote.mlp import SimpleMLP as _BNSimple
            network = _BNSimple()
        else:
            raise ValueError(f"Unknown network '{network_name}' for dataset '{dataset_name}'")

    elif dataset_name == "DryBean":
        print(f'Loading network {network_name} for DryBean ...')
        if network_name == "BeanMLP":
            from dlModels.DryBean.mlp import BeanMLP
            network = BeanMLP()
        else:
            raise ValueError(f"Unknown network '{network_name}' for dataset '{dataset_name}'")

    elif dataset_name == 'Iris':
        print(f'Loading network {network_name} for Iris ...')
        if network_name == 'MiniMLP3':
            from dlModels.Iris.mlp import MiniMLP3
            network = MiniMLP3()
        else:
            raise ValueError(f"Unknown network '{network_name}' for dataset '{dataset_name}'")

    elif dataset_name == 'Wine':
        print(f'Loading network {network_name} for Wine ...')
        if network_name == 'WineMLP':
            from dlModels.Wine.mlp import WineMLP
            network = WineMLP()
        else:
            raise ValueError(f"Unknown network '{network_name}' for dataset '{dataset_name}'")

    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'")

    network.to(device)
    network.eval()
    return network


# ============================== Dataloaders ==============================

def get_loader(network_name: str,
               batch_size: int,
               image_per_class: int = None,
               dataset_name: str = None,
               network: torch.nn.Module = None) -> tuple:
    """
    Ritorna i loader (train, val, test) per il dataset richiesto.
    """
    if dataset_name == 'Letter':
        print('Loading Letter dataset...')
        return load_letter_dataset(batch_size=batch_size)

    if dataset_name == 'Banknote':
        print('Loading Banknote dataset...')
        return load_banknote_dataset(batch_size=batch_size)

    if dataset_name == 'Wine':
        print('Loading Wine dataset...')
        return load_wine_dataset(batch_size=batch_size)

    if dataset_name == 'DryBean':
        print('Loading DryBean dataset...')
        return load_drybean_dataset(batch_size=batch_size)

    if dataset_name == 'Iris':
        print('Loading Iris dataset...')
        return load_iris_dataset(batch_size=batch_size)

    if dataset_name == 'BreastCancer':
        print('Loading BreastCancer dataset...')
        return load_breastCancer_datasets(
            train_batch_size=batch_size,
            test_batch_size=batch_size
        )

    if dataset_name == 'CIFAR10':
        print('Loading CIFAR10 dataset...')
        train_loader, _, test_loader = load_CIFAR10_datasets(
            test_batch_size=batch_size,
            test_image_per_class=image_per_class
        )
        return train_loader, None, test_loader

    if dataset_name == 'CIFAR100':
        print('Loading CIFAR100 dataset...')
        train_loader, _, test_loader = Load_CIFAR100_datasets(
            test_batch_size=batch_size,
            test_image_per_class=image_per_class
        )
        return train_loader, None, test_loader

    if dataset_name == 'GTSRB':
        print('Loading GTSRB dataset...')
        train_loader, _, test_loader = Load_GTSRB_datasets(
            test_batch_size=batch_size,
            test_image_per_class=image_per_class
        )
        return train_loader, None, test_loader

    if network_name in ['SimpleMLP', 'BiggerMLP']:
        print(f"Loading BreastCancer dataset for {network_name}...")
        return load_breastCancer_datasets(
            train_batch_size=batch_size,
            test_batch_size=batch_size
        )

    raise ValueError(f"Dataset/network '{dataset_name or network_name}' non riconosciuto")


# ============================== Module helpers (CNN) ==============================

def get_delayed_start_module(network: Module, network_name: str) -> Module:
    if 'LeNet' in network_name:
        delayed_start_module = network
    elif 'ResNet' in network_name:
        delayed_start_module = network
    elif 'MobileNetV2' in network_name:
        delayed_start_module = network.features
        print('delayed_start_module:', delayed_start_module)
    elif 'DenseNet' in network_name:
        delayed_start_module = network.features
    elif 'EfficientNet' in network_name:
        delayed_start_module = network.features
    else:
        raise UnknownNetworkException
    return delayed_start_module


def get_module_classes(network_name: str) -> Union[List[type], type]:
    if 'LeNet' in network_name:
        module_classes = Sequential
    elif 'MobileNetV2' in network_name:
        module_classes = Sequential
    elif 'ResNet' in network_name:
        if network_name in ['ResNet18', 'ResNet50']:
            module_classes = Sequential
        else:
            module_classes = resnet.BasicBlock
    elif 'DenseNet' in network_name:
        module_classes = (_DenseBlock, _Transition)
    elif 'EfficientNet' in network_name:
        module_classes = (Conv2dNormActivation, Conv2dNormActivation)
    else:
        raise UnknownNetworkException(f'Unknown network {network_name}')
    return module_classes


# ============================== Fault list / device ==============================

def get_fault_list(fault_model: str,
                   fault_list_generator: FLManager,
                   e: float = .01,
                   t: float = 2.58) -> Tuple[Union[List[NeuronFault], List[WeightFault]], List[Module]]:
    if fault_model == 'byzantine_neuron':
        fault_list = fault_list_generator.get_neuron_fault_list()
    elif fault_model in ('stuck-at_params', 'bit-flip'):
        fault_list = fault_list_generator.get_weight_fault_list()
    else:
        raise ValueError(f'Invalid fault model {fault_model}')
    injectable_modules = fault_list_generator.injectable_output_modules_list
    return fault_list, injectable_modules


def get_device(use_cuda0: bool, use_cuda1: bool) -> torch.device:
    if use_cuda0:
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            print('ERROR: cuda:0 not available even if use-cuda is set')
            exit(-1)
    elif use_cuda1:
        if torch.cuda.is_available():
            device = 'cuda:1'
        else:
            print('ERROR: cuda:1 not available even if use-cuda is set')
            exit(-1)
    else:
        device = 'cpu'
    return torch.device(device)


# ============================== CSV print for FI ==============================

def formatted_print(fault_list: list,
                    network_name: str,
                    batch_size: int,
                    batch_id: int,
                    faulty_prediction_dict: dict,
                    fault_dropping: bool = False,
                    fault_delayed_start: bool = False) -> None:

    fault_list_rows = [[fault_id,
                        fault.layer_name,
                        fault.tensor_index[0],
                        fault.tensor_index[1] if len(fault.tensor_index) > 1 else np.nan,
                        fault.tensor_index[2] if len(fault.tensor_index) > 2 else np.nan,
                        fault.tensor_index[3] if len(fault.tensor_index) > 3 else np.nan,
                        fault.bit,
                        fault.value]
                       for fault_id, fault in enumerate(fault_list)]

    fault_list_columns = [
        'Fault_ID', 'Fault_Layer',
        'Fault_Index_0', 'Fault_Index_1', 'Fault_Index_2', 'Fault_Index_3',
        'Fault_Bit', 'Fault_Value'
    ]

    prediction_rows = [
        [fault_id, batch_id, prediction_id, prediction[0], prediction[1]]
        for fault_id in faulty_prediction_dict
        for prediction_id, prediction in enumerate(faulty_prediction_dict[fault_id])
    ]

    prediction_columns = ['Fault_ID', 'Batch_ID', 'Image_ID', 'Top_1', 'Top_Score']

    fault_list_df = pd.DataFrame(fault_list_rows, columns=fault_list_columns)
    prediction_df = pd.DataFrame(prediction_rows, columns=prediction_columns)
    complete_df = fault_list_df.merge(prediction_df, on='Fault_ID')

    file_prefix = 'combined_' if fault_dropping and fault_delayed_start \
        else 'delayed_' if fault_delayed_start \
        else 'dropping_' if fault_dropping \
        else ''

    output_folder = f'output/fault_campaign_results/{network_name}/{batch_size}'
    os.makedirs(output_folder, exist_ok=True)
    complete_df.to_csv(f'{output_folder}/{file_prefix}fault_injection_batch_{batch_id}.csv', index=False)


# ============================== Dataset loaders (tabular) ==============================

def load_letter_dataset(batch_size=64):
    """
    Carica il dataset Letter Recognition da CSV.
    Usa subset (A–G) per ridurre classi e dimensione.
    """
    dataset_path = os.path.join("Datasets", "Letter", "letter-recognition.csv")
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"File non trovato: {dataset_path}")

    df = pd.read_csv(dataset_path, header=None)
    df.columns = ["letter"] + [f"f{i}" for i in range(1, 17)]
    df["letter"] = df["letter"].apply(lambda x: ord(x) - ord("A"))
    df = df[df["letter"] <= 6]

    # sottocampionamento stratificato
    df_small = df.groupby("letter", group_keys=False).apply(
        lambda x: x.sample(n=200, random_state=42)
    )

    X = df_small.drop("letter", axis=1).values
    y = df_small["letter"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1, random_state=42, stratify=y_trainval
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    print("Letter dataset loaded (subset of 7 classes, ~1000 samples).")
    return train_loader, val_loader, test_loader


def load_breastCancer_datasets(train_batch_size=32, train_split=0.8, test_batch_size=1, test_image_per_class=None):
    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_split_length = int(len(train_dataset) * train_split)
    val_split_length = len(train_dataset) - train_split_length
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset,
        lengths=[train_split_length, val_split_length],
        generator=torch.Generator().manual_seed(1234)
    )

    train_loader = DataLoader(dataset=train_subset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_subset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)

    print('Breast Cancer Dataset loaded')
    return train_loader, val_loader, test_loader


def load_banknote_dataset(batch_size=32):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
    column_names = ["variance", "skewness", "curtosis", "entropy", "class"]
    df = pd.read_csv(url, header=None, names=column_names)

    X = df.drop("class", axis=1).values
    y = df["class"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_scaled, y, train_size=0.7, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1, random_state=42, stratify=y_trainval
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def load_drybean_dataset(batch_size=64, total_samples=1400):
    path = os.path.join("dlModels", "DryBean", "Dry_Bean_Dataset.arff")
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)

    if isinstance(df["Class"][0], bytes):
        df["Class"] = df["Class"].str.decode("utf-8")

    # downsampling stratificato
    df_small = df.groupby("Class", group_keys=False).apply(
        lambda x: x.sample(frac=total_samples / len(df), random_state=42)
    ).reset_index(drop=True)

    X = df_small.drop("Class", axis=1).values.astype("float32")
    y = LabelEncoder().fit_transform(df_small["Class"])

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=42)

    def make_loader(X, y, shuffle=True):
        return DataLoader(
            TensorDataset(torch.tensor(X), torch.tensor(y, dtype=torch.long)),
            batch_size=batch_size, shuffle=shuffle
        )

    return (
        make_loader(X_train, y_train, shuffle=True),
        make_loader(X_val, y_val, shuffle=True),
        make_loader(X_test, y_test, shuffle=False)
    )


def load_iris_dataset(batch_size=32):
    iris = load_iris()
    X = iris.data
    y = iris.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1, random_state=42, stratify=y_trainval
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def load_wine_dataset(batch_size=32):
    data = load_wine()
    X = data.data
    y = data.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.1, random_state=42, stratify=y_trainval
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# ============================== State dict loader ==============================

def load_from_dict(network, device, path, function=None):
    if '.th' in path:
        state_dict = torch.load(path, map_location=device)['state_dict']
        print('state_dict loaded')
    else:
        state_dict = torch.load(path, map_location=device)
        print('state_dict loaded')

    if function is None:
        clean_state_dict = {key.replace('module.', ''): value
                            for key, value in state_dict.items()}
    else:
        clean_state_dict = {
            key.replace('module.', ''):
                function(value) if not (('bn' in key) and ('weight' in key)) else value
            for key, value in state_dict.items()
        }

    network.load_state_dict(clean_state_dict, strict=False)
    print('state_dict loaded into network')


# ============================== FS helpers ==============================

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        print(f"Creating missing directory: {directory}")
        os.makedirs(directory)


def ensure_file_exists(file_path, default_data=None):
    if not os.path.exists(file_path):
        print(f"Creating missing file: {file_path}")
        if default_data is not None:
            np.save(file_path, default_data)
        else:
            open(file_path, 'w').close()


def count_batch(folder, path):
    try:
        if not os.path.exists(folder):
            print(f"Error: Directory {folder} does not exist.")
            return 0, 0, 0

        files = os.listdir(folder)
        if not files:
            print(f"Warning: No files found in {folder}.")
            return 0, 0, 0

        loaded_file = np.load(path, allow_pickle=True)
        if loaded_file.size == 0:
            print(f"Warning: {path} is empty. Skipping.")
            return 0, 0, 0

        n_outputs = loaded_file.shape[2]
        n_faults = loaded_file.shape[0]
        return len(files), n_outputs, n_faults

    except FileNotFoundError:
        print(f"Error: File not found {path}.")
        return 0, 0, 0
    except EOFError:
        print(f"Error: No data left in file {path}. Skipping.")
        return 0, 0, 0
    except Exception as e:
        print(f"Unexpected error in count_batch: {e}")
        return 0, 0, 0


# ============================== FI analysis (single / parallel / chunked) ==============================

def output_definition(test_loader, batch_size):
    clean_output_path = os.path.join(SETTINGS.CLEAN_OUTPUT_FOLDER, 'clean_output.npy')
    if not os.path.exists(clean_output_path):
        print(f" ERROR: Clean output file {clean_output_path} is missing!")
        return

    print(" Caricamento output clean...")
    loaded_clean_output = np.load(clean_output_path, allow_pickle=True)
    number_of_clean_batches = len(loaded_clean_output)

    batch_folder = os.path.join(SETTINGS.FAULTY_OUTPUT_FOLDER, SETTINGS.FAULT_MODEL)
    batch_path = os.path.join(batch_folder, 'batch_0.npy')
    number_of_batch, _, max_faults = count_batch(batch_folder, batch_path)
    number_of_batch = min(number_of_batch, number_of_clean_batches)
    n_faults = max_faults if SETTINGS.FAULTS_TO_INJECT == -1 else SETTINGS.FAULTS_TO_INJECT

    output_path = os.path.join(SETTINGS.FI_ANALYSIS_PATH, "output_analysis.csv")
    os.makedirs(SETTINGS.FI_ANALYSIS_PATH, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Fault_ID', 'batch', 'image', 'output'])

        for fault_id in tqdm(range(n_faults), desc="Analysis", colour='cyan'):
            for i in range(number_of_batch):
                path = os.path.join(batch_folder, f'batch_{i}.npy')
                if not os.path.exists(path):
                    continue

                faulty = np.load(path, allow_pickle=True)
                if fault_id >= faulty.shape[0]:
                    continue

                clean_batch = loaded_clean_output[i]
                current_batch_size = min(len(clean_batch), batch_size)

                clean_top = np.argmax(clean_batch, axis=1)
                faulty_top = np.argmax(faulty[fault_id], axis=1)
                same_label = (clean_top == faulty_top)

                clean_conf = clean_batch[np.arange(current_batch_size), clean_top]
                faulty_conf = faulty[fault_id][np.arange(current_batch_size), clean_top]
                delta = np.abs(faulty_conf - clean_conf) / np.maximum(1e-8, np.abs(clean_conf))

                masked = np.all(clean_batch == faulty[fault_id], axis=1)

                outputs = np.full(current_batch_size, 4, dtype=int)
                outputs[masked] = 0
                outputs[same_label & (delta < 0.1)] = 1
                outputs[same_label & (delta >= 0.1) & (delta < 0.2)] = 2
                outputs[same_label & (delta >= 0.2)] = 3

                for j in range(current_batch_size):
                    writer.writerow([fault_id, i, j, int(outputs[j])])

    print("\nOutput analysis completata.")


from concurrent.futures import ProcessPoolExecutor, as_completed


def _analyze_fault_range(start_id, end_id, batch_folder, clean_output_path, batch_size, n_batches):
    results = []
    clean_output = np.load(clean_output_path, allow_pickle=True)

    for fault_id in range(start_id, end_id):
        for i in range(n_batches):
            path = os.path.join(batch_folder, f'batch_{i}.npy')
            if not os.path.exists(path):
                continue

            faulty = np.load(path, allow_pickle=True)
            if fault_id >= faulty.shape[0]:
                continue

            clean_batch = clean_output[i]
            for j in range(min(len(clean_batch), batch_size)):
                clean = clean_batch[j]
                faulty_out = faulty[fault_id, j]
                clean_top = np.argmax(clean)
                faulty_top = np.argmax(faulty_out)
                delta = abs(faulty_out[clean_top] - clean[clean_top]) / max(1e-8, abs(clean[clean_top]))

                if np.array_equal(clean, faulty_out):
                    results.append([fault_id, i, j, 0])
                elif clean_top == faulty_top:
                    if delta >= 0.2:
                        results.append([fault_id, i, j, 3])
                    elif delta >= 0.1:
                        results.append([fault_id, i, j, 2])
                    else:
                        results.append([fault_id, i, j, 1])
                else:
                    results.append([fault_id, i, j, 4])
    return results


def _analyze_fault_range_star(args):
    return _analyze_fault_range(*args)


def output_definition_parallel(test_loader, batch_size, n_workers=32):
    clean_output_path = os.path.join(SETTINGS.CLEAN_OUTPUT_FOLDER, 'clean_output.npy')
    if not os.path.exists(clean_output_path):
        print(f" ERROR: Clean output file {clean_output_path} is missing!")
        return

    batch_folder = os.path.join(SETTINGS.FAULTY_OUTPUT_FOLDER, SETTINGS.FAULT_MODEL)
    batch_path = os.path.join(batch_folder, 'batch_0.npy')
    number_of_batch, _, max_faults = count_batch(batch_folder, batch_path)

    loaded_clean_output = np.load(clean_output_path, allow_pickle=True)
    number_of_batch = min(number_of_batch, len(loaded_clean_output))
    n_faults = max_faults if SETTINGS.FAULTS_TO_INJECT == -1 else SETTINGS.FAULTS_TO_INJECT

    print(f" Parallel analysis with {n_workers} workers on {n_faults} faults × {number_of_batch} batches")

    os.makedirs(SETTINGS.FI_ANALYSIS_PATH, exist_ok=True)
    output_path = os.path.join(SETTINGS.FI_ANALYSIS_PATH, "output_analysis.csv")

    chunk_size = math.ceil(n_faults / n_workers)
    chunks = [(i, min(i + chunk_size, n_faults)) for i in range(0, n_faults, chunk_size)]
    args = [(start, end, batch_folder, clean_output_path, batch_size, number_of_batch) for (start, end) in chunks]

    all_results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_analyze_fault_range_star, arg) for arg in args]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Parallel Fault Analysis"):
            all_results.extend(future.result())

    print(f" Saving analysis to {output_path}...")
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Fault_ID', 'batch', 'image', 'output'])
        writer.writerows(all_results)

    print(" Output analysis completata e salvata.")


# ---- Chunked version ----

def _init_clean_output(path):
    global CLEAN_OUTPUT
    CLEAN_OUTPUT = np.load(path, allow_pickle=True)


def _analyze_fault_range_chunk(chunk_id, fault_ids, batch_folder, batch_size, n_batches, output_dir):
    global CLEAN_OUTPUT
    results = []

    print(f"[Chunk {chunk_id}] Analisi fault IDs: {fault_ids[:5]}...")

    for fault_id in fault_ids:
        for i in range(n_batches):
            path = os.path.join(batch_folder, f'batch_{i}.npy')
            if not os.path.exists(path):
                continue

            faulty = np.load(path, allow_pickle=True, mmap_mode='r')
            if fault_id >= faulty.shape[0]:
                continue

            clean_batch = CLEAN_OUTPUT[i]
            for j in range(min(len(clean_batch), batch_size)):
                clean = clean_batch[j]
                faulty_out = faulty[fault_id, j]
                clean_top = np.argmax(clean)
                faulty_top = np.argmax(faulty_out)
                delta = abs(faulty_out[clean_top] - clean[clean_top]) / max(1e-8, abs(clean[clean_top]))

                if np.array_equal(clean, faulty_out):
                    results.append([fault_id, i, j, 0])
                elif clean_top == faulty_top:
                    if delta >= 0.2:
                        results.append([fault_id, i, j, 3])
                    elif delta >= 0.1:
                        results.append([fault_id, i, j, 2])
                    else:
                        results.append([fault_id, i, j, 1])
                else:
                    results.append([fault_id, i, j, 4])

    chunk_file = os.path.join(output_dir, f"output_analysis_part_{chunk_id}.csv")
    with open(chunk_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Fault_ID', 'batch', 'image', 'output'])
        writer.writerows(results)

    print(f"[Chunk {chunk_id}] Completato. Salvato in {chunk_file}")
    return chunk_file


def _analyze_fault_range_chunk_star(args):
    return _analyze_fault_range_chunk(*args)


def output_definition_parallel_chunked(test_loader, batch_size, n_workers=32, chunk_size=5000):
    clean_output_path = os.path.join(SETTINGS.CLEAN_OUTPUT_FOLDER, 'clean_output.npy')
    if not os.path.exists(clean_output_path):
        print(f" Clean output not found: {clean_output_path}")
        return

    print(f"Loaded CLEAN_OUTPUT with shape: {np.load(clean_output_path, allow_pickle=True).shape}")

    batch_folder = os.path.join(SETTINGS.FAULTY_OUTPUT_FOLDER, SETTINGS.FAULT_MODEL)
    batch_path = os.path.join(batch_folder, 'batch_0.npy')
    number_of_batch, _, max_faults = count_batch(batch_folder, batch_path)

    n_faults = max_faults if SETTINGS.FAULTS_TO_INJECT == -1 else SETTINGS.FAULTS_TO_INJECT
    output_dir = SETTINGS.FI_ANALYSIS_PATH
    os.makedirs(output_dir, exist_ok=True)

    fault_ids = list(range(n_faults))
    chunks = [fault_ids[i:i + chunk_size] for i in range(0, n_faults, chunk_size)]

    args = []
    for chunk_id, fault_chunk in enumerate(chunks):
        chunk_file = os.path.join(output_dir, f"output_analysis_part_{chunk_id}.csv")
        if os.path.exists(chunk_file):
            print(f" Chunk {chunk_id} already exists, skipping.")
            continue
        args.append((chunk_id, fault_chunk, batch_folder, batch_size, number_of_batch, output_dir))

    print(f" Avvio analisi parallela in {len(args)} chunk da {chunk_size} fault, usando {n_workers} core...")

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_clean_output, initargs=(clean_output_path,)) as executor:
        for i, _ in enumerate(executor.map(_analyze_fault_range_chunk_star, args)):
            print(f"Chunk {i + 1}/{len(args)} completato")

    # Merge finale
    merged_path = os.path.join(output_dir, "output_analysis.csv")
    with open(merged_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Fault_ID', 'batch', 'image', 'output'])
        for chunk_id in range(len(chunks)):
            chunk_file = os.path.join(output_dir, f"output_analysis_part_{chunk_id}.csv")
            if os.path.exists(chunk_file):
                with open(chunk_file, 'r') as infile:
                    next(infile)  # Skip header
                    for row in infile:
                        outfile.write(row)
    print(f" Analisi completata. File finale salvato in {merged_path}")


# ============================== CSV summary (single / parallel / chunked) ==============================

def csv_summary():
    FI_ANALYSIS_PATH = SETTINGS.FI_ANALYSIS_PATH
    FAULT_LIST_PATH = os.path.join(SETTINGS.FAULT_LIST_PATH, SETTINGS.FAULT_LIST_NAME)
    OUTPUT_FILE_PATH = SETTINGS.FI_SUM_ANALYSIS_PATH

    print(f"FI_ANALYSIS_PATH: {FI_ANALYSIS_PATH}")
    print(f"Fault list path: {FAULT_LIST_PATH}")
    print(f"Summary output path: {OUTPUT_FILE_PATH}")

    os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)

    fault_df = pd.read_csv(FAULT_LIST_PATH, usecols=["Injection", "Layer", "TensorIndex", "Bit"])
    print(" Fault list loaded.")

    output_analysis_path = os.path.join(FI_ANALYSIS_PATH, "output_analysis.csv")
    df = pd.read_csv(output_analysis_path)

    if df['output'].dtype != int:
        df['output'] = df['output'].astype(int)

    summary_rows = []
    for injection_id, group in fault_df.groupby("Injection"):
        subset = df[df['Fault_ID'] == injection_id]

        masked = (subset['output'] == 0).sum()
        non_critical = subset['output'].isin([1, 2, 3]).sum()
        critical = (subset['output'] == 4).sum()
        total = masked + non_critical + critical

        acc = (masked + non_critical) / total if total > 0 else 0.0
        fr = critical / total if total > 0 else 0.0

        summary_rows.append({
            "Injection": injection_id,
            "Layers": list(group["Layer"].unique()),
            "TensorIndices": list(group["TensorIndex"].unique()),
            "Bits": list(group["Bit"].unique()),
            "masked": int(masked),
            "non_critical": int(non_critical),
            "critical": int(critical),
            "accuracy": round(acc, 4),
            "failure_rate": round(fr, 4)
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_FILE_PATH, index=False)
    print(f" Summary CSV saved to {OUTPUT_FILE_PATH}")


def save_global_metrics_summary_txt():
    csv_path = SETTINGS.FI_SUM_ANALYSIS_PATH
    if not os.path.exists(csv_path):
        print(f"Summary CSV not found: {csv_path}")
        return

    from pandas.errors import EmptyDataError
    try:
        summary_iter = pd.read_csv(csv_path, chunksize=10000)
    except EmptyDataError:
        print(f"[WARNING] {csv_path} is empty or has no columns. Skipping summary generation.")
        return

    total = masked = non_critical = critical = 0
    for chunk in summary_iter:
        masked += chunk['masked'].sum()
        non_critical += chunk['non_critical'].sum()
        critical += chunk['critical'].sum()
        total += (chunk['masked'] + chunk['non_critical'] + chunk['critical']).sum()

    acc = (masked + non_critical) / total if total > 0 else 0
    fr = critical / total if total > 0 else 0

    fault_bits = SETTINGS.NUM_FAULTS_TO_INJECT
    model = SETTINGS.NETWORK
    dataset = SETTINGS.DATASET
    txt_path = os.path.join(SETTINGS.FI_ANALYSIS_PATH, f"N{fault_bits}_{dataset}_{model}_summary.txt")

    with open(txt_path, 'w') as f:
        f.write(f"NUM_FAULTS_TO_INJECT: {fault_bits}\n")
        f.write(f"Total faults: {total}\n")
        f.write(f"Masked: {masked}\n")
        f.write(f"NonCritical: {non_critical}\n")
        f.write(f"Critical: {critical}\n")
        f.write(f"Accuracy (masked + non-critical): {acc:.4f}\n")
        f.write(f"Failure Rate (critical): {fr:.4f}\n")

    print(f" Summary TXT saved to: {txt_path}")


def _process_injection(inj_id, group_df, df):
    fault_outputs = df[df['Fault_ID'] == inj_id]['output']

    masked = (fault_outputs == 0).sum()
    non_critical = fault_outputs.isin([1, 2, 3]).sum()
    critical = (fault_outputs == 4).sum()

    layers = group_df['Layer'].tolist()
    indices = group_df['TensorIndex'].tolist()
    bits = group_df['Bit'].tolist()

    total = masked + non_critical + critical
    accuracy = (masked + non_critical) / total if total > 0 else 0.0
    failure_rate = critical / total if total > 0 else 0.0

    return {
        'Injection': inj_id,
        'Layers': str(layers),
        'TensorIndices': str(indices),
        'Bits': str(bits),
        'masked': masked,
        'non_critical': non_critical,
        'critical': critical,
        'accuracy': round(accuracy, 4),
        'failure_rate': round(failure_rate, 4)
    }


def _process_injection_star(args):
    inj_id, group_df, df = args
    return _process_injection(inj_id, group_df, df)


def csv_summary_parallel(n_workers=32):
    input_file_path = f'{SETTINGS.FI_ANALYSIS_PATH}/output_analysis.csv'
    fault_list_path = f'{SETTINGS.FAULT_LIST_PATH}/{SETTINGS.FAULT_LIST_NAME}'
    output_file_path = f'{SETTINGS.FI_SUM_ANALYSIS_PATH}'

    print(f"FI_ANALYSIS_PATH: {SETTINGS.FI_ANALYSIS_PATH}")
    print(f"Fault list path: {SETTINGS.FAULT_LIST_PATH}")
    print(f"Summary output path: {output_file_path}")

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    try:
        df = pd.read_csv(input_file_path)
        print(' Output analysis loaded.')
    except FileNotFoundError:
        print(f" File not found: {input_file_path}")
        return

    try:
        fault_df = pd.read_csv(fault_list_path)
        print(' Fault list loaded.')
    except FileNotFoundError:
        print(f" File not found: {fault_list_path}")
        return

    grouped = fault_df.groupby("Injection")
    args = [(inj_id, group.copy(), df) for inj_id, group in grouped]

    print(f" Parallelizing summary on {n_workers} cores...")
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        output_rows = list(tqdm(executor.map(_process_injection_star, args),
                                total=len(args),
                                desc="Calculating summary"))

    summary_df = pd.DataFrame(output_rows)
    summary_df.to_csv(output_file_path, index=False)
    print(f" Summary CSV saved to {output_file_path}")


def _process_summary_chunk(chunk_id, injection_ids, fault_df_path, analysis_csv_path, output_dir):
    df = pd.read_csv(analysis_csv_path)
    fault_df = pd.read_csv(fault_df_path)

    fault_outputs_dict = df.groupby("Fault_ID")["output"].apply(list).to_dict()
    grouped = fault_df.groupby("Injection")

    output_rows = []
    for inj_id in injection_ids:
        if inj_id not in grouped.groups:
            continue
        group_df = grouped.get_group(inj_id)
        outputs = fault_outputs_dict.get(inj_id, [])

        masked = sum(1 for o in outputs if o == 0)
        non_critical = sum(1 for o in outputs if o in [1, 2, 3])
        critical = sum(1 for o in outputs if o == 4)

        layers = group_df['Layer'].tolist()
        indices = group_df['TensorIndex'].tolist()
        bits = group_df['Bit'].tolist()

        total = masked + non_critical + critical
        accuracy = (masked + non_critical) / total if total > 0 else 0.0
        failure_rate = critical / total if total > 0 else 0.0

        output_rows.append({
            'Injection': inj_id,
            'Layers': str(layers),
            'TensorIndices': str(indices),
            'Bits': str(bits),
            'masked': masked,
            'non_critical': non_critical,
            'critical': critical,
            'accuracy': round(accuracy, 4),
            'failure_rate': round(failure_rate, 4)
        })

    chunk_path = os.path.join(output_dir, f"summary_chunk_{chunk_id}.csv")
    pd.DataFrame(output_rows).to_csv(chunk_path, index=False)
    return chunk_path


def _process_summary_chunk_star(args):
    return _process_summary_chunk(*args)


def csv_summary_parallel_chunked(n_workers=32, chunk_size=5000):
    analysis_csv_path = os.path.join(SETTINGS.FI_ANALYSIS_PATH, "output_analysis.csv")
    fault_df_path = os.path.join(SETTINGS.FAULT_LIST_PATH, SETTINGS.FAULT_LIST_NAME)
    final_output_path = SETTINGS.FI_SUM_ANALYSIS_PATH
    output_dir = os.path.dirname(final_output_path)

    os.makedirs(output_dir, exist_ok=True)

    try:
        fault_df = pd.read_csv(fault_df_path)
        print(" Fault list loaded.")
    except FileNotFoundError:
        print(f" File not found: {fault_df_path}")
        return

    injection_ids = fault_df["Injection"].unique()
    chunks = [injection_ids[i:i + chunk_size] for i in range(0, len(injection_ids), chunk_size)]
    args = [(i, list(chunk), fault_df_path, analysis_csv_path, output_dir) for i, chunk in enumerate(chunks)]

    print(f"🧠 Avvio sintesi CSV su {len(chunks)} chunk da {chunk_size} con {n_workers} worker...")
    chunk_files = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        for chunk_path in tqdm(executor.map(_process_summary_chunk_star, args),
                               total=len(args), desc="Calculating chunked summary"):
            chunk_files.append(chunk_path)

    summary_df = pd.concat([pd.read_csv(f) for f in chunk_files], ignore_index=True)
    summary_df.to_csv(final_output_path, index=False)
    print(f" Summary CSV finale salvato in {final_output_path}")


# ============================== Fault list generator ==============================

def fault_list_gen():
    """
    Genera una fault list esaustiva (o random) di singoli bit-flip sui pesi della rete FLOAT.
    Usa SETTINGS.NETWORK / SETTINGS.DATASET (coerenza nomi).
    """
    random.seed(SETTINGS.SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    network = get_network(
        network_name=SETTINGS.NETWORK,
        device=device,
        dataset_name=SETTINGS.DATASET
    )
    network.to(device).eval()

    bit_width = 8  # 8-bit
    bit_positions = list(range(bit_width))
    all_bit_faults = []

    # 1) singoli bit flip possibili
    intermediate_singles_path = f"{SETTINGS.FAULT_LIST_PATH}/single_faults.csv"
    os.makedirs(SETTINGS.FAULT_LIST_PATH, exist_ok=True)
    with open(intermediate_singles_path, 'w', newline='') as single_file:
        writer = csv.writer(single_file)
        writer.writerow(['Injection', 'Layer', 'TensorIndex', 'Bit'])
        inj_id = 0
        for name, param in network.named_parameters():
            if 'weight' in name:
                layer_name = name.replace('.weight', '')
                shape = param.shape
                for idx in np.ndindex(*shape):
                    for bit in bit_positions:
                        all_bit_faults.append((layer_name, idx, bit))
                        writer.writerow([inj_id, layer_name, idx, bit])
                        inj_id += 1

    print(f" Salvati {len(all_bit_faults)} bit flip singoli in {intermediate_singles_path}")

    # 2) combinazioni da NUM_FAULTS_TO_INJECT
    import itertools
    intermediate_combos_path = f"{SETTINGS.FAULT_LIST_PATH}/combinations_{SETTINGS.NUM_FAULTS_TO_INJECT}.csv"
    combinations = list(itertools.combinations(all_bit_faults, SETTINGS.NUM_FAULTS_TO_INJECT))

    with open(intermediate_combos_path, 'w', newline='') as combo_file:
        writer = csv.writer(combo_file)
        writer.writerow(['GroupID'] + [f"Fault{i+1}" for i in range(SETTINGS.NUM_FAULTS_TO_INJECT)])
        for i, combo in enumerate(combinations):
            row = [i] + [f"{layer},{idx},{bit}" for (layer, idx, bit) in combo]
            writer.writerow(row)

    print(f" Salvate {len(combinations)} combinazioni di {SETTINGS.NUM_FAULTS_TO_INJECT} bit in {intermediate_combos_path}")

    # 3) selezione finale
    if SETTINGS.FAULTS_TO_INJECT == -1:
        selected = combinations
        print(f" Fault list ESAUSTIVA: {len(selected)} combinazioni da {SETTINGS.NUM_FAULTS_TO_INJECT} bit flip.")
    else:
        selected = random.sample(combinations, SETTINGS.FAULTS_TO_INJECT)
        print(f" Fault list RANDOM: {SETTINGS.FAULTS_TO_INJECT} combinazioni da {SETTINGS.NUM_FAULTS_TO_INJECT} bit flip.")

    # 4) scrittura CSV finale
    final_csv_path = f"{SETTINGS.FAULT_LIST_PATH}/{SETTINGS.FAULT_LIST_NAME}"
    with open(final_csv_path, 'w', newline='') as final_file:
        writer = csv.writer(final_file)
        writer.writerow(['Injection', 'Layer', 'TensorIndex', 'Bit'])
        for inj_id, combo in enumerate(selected):
            for layer, idx, bit in combo:
                writer.writerow([inj_id, layer, str(idx), bit])

    print(f" Fault list finale scritta in {final_csv_path} con {len(selected)} iniezioni.")


# ============================== Sampling faults ==============================

def num_experiments_needed(p_estimate=0.5):
    e = SETTINGS.error_margin
    t = SETTINGS.confidence_constant
    n_exp = int((t ** 2 * p_estimate * (1 - p_estimate)) / (e ** 2))
    print(f"Numero minimo di esperimenti necessari: {n_exp}")
    return n_exp


def select_random_faults(fault_list_path, num_faults_needed):
    fault_df = pd.read_csv(fault_list_path)
    sampled_faults = fault_df.sample(n=num_faults_needed, random_state=SETTINGS.SEED)
    sampled_faults.to_csv(fault_list_path.replace('.csv', '_sampled.csv'), index=False)
    print(f"Fault casualmente selezionati salvati in {fault_list_path.replace('.csv', '_sampled.csv')}")
