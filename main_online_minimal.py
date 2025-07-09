import torch
import time
from itertools import product
from tqdm import tqdm
from faultManager.WeightFault import WeightFault
from faultManager.WeightFaultInjector import WeightFaultInjector
from utils import get_loader
import heapq
from torch.quantization import QuantWrapper, get_default_qconfig
from dlModels.Banknote.mlp import SimpleMLP
import random

def _build_fault_list(num_faults_to_inject, faults_to_inject, model):
    all_faults = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.quantized.Linear):
            weight, _ = module._packed_params._weight_bias()
            shape = weight.shape
            for idx in product(*[range(s) for s in shape]):
                for bit in range(8):
                    all_faults.append((name, idx, bit))
    print(f"Total single faults available: {len(all_faults)}")
    print(f"Running random campaign with {faults_to_inject} injections of {num_faults_to_inject} simultaneous faults")
    final_combinations = set()
    while len(final_combinations) < faults_to_inject:
        combo = tuple(sorted(random.sample(all_faults, num_faults_to_inject)))
        final_combinations.add(combo)
    fault_combinations = list(final_combinations)
    random.shuffle(fault_combinations)
    return fault_combinations

def run_online_fault_injection_minimal(num_faults_to_inject, fault_combinations_or_num, batch_idx=None):
    """
    fault_combinations_or_num può essere:
        - una lista di tuple (batch), usata per N grandi e batching
        - un int: n campioni random, come prima
    """
    start_time = time.time()
    device = torch.device("cpu")
    print(f"Using device: {device}")

    print("Loading raw model and applying quantization manually...")
    base_model = SimpleMLP()
    base_model.qconfig = get_default_qconfig("fbgemm")
    model = QuantWrapper(base_model)
    model.quantize_model = base_model.quantize_model
    model.to(device)

    _, _, test_loader = get_loader("SimpleMLP", 64, dataset_name="Banknote")
    print("Test loader ready.")

    if hasattr(model, 'quantize_model') and callable(model.quantize_model):
        print("Applying quantization using model.quantize_model...")
        model.quantize_model(calib_loader=test_loader)
        print("Quantization completed.")
    else:
        print("WARNING: model.quantize_model not found. Skipping quantization.")

    print("Computing clean predictions...")
    clean_predictions = []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1)
            clean_predictions.append(preds.cpu())
    clean_predictions = torch.cat(clean_predictions)

    # Ricostruisci sempre all_faults
    all_faults = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.quantized.Linear):
            weight, _ = module._packed_params._weight_bias()
            shape = weight.shape
            for idx in product(*[range(s) for s in shape]):
                for bit in range(8):
                    all_faults.append((name, idx, bit))

    # --- Gestione combinazioni: batch fornito oppure random come prima ---
    if isinstance(fault_combinations_or_num, int):
        faults_to_inject = fault_combinations_or_num
        fault_combinations = _build_fault_list(num_faults_to_inject, faults_to_inject, model)
    else:
        fault_combinations = fault_combinations_or_num
        faults_to_inject = len(fault_combinations)
        print(f"Received {faults_to_inject} precomputed fault combinations for N={num_faults_to_inject}")

    batch_str = f"_batch{batch_idx}" if batch_idx is not None else ""
    output_path = f"top100_faults_progress_N{num_faults_to_inject}{batch_str}.txt"
    output_path_final = f"top100_faults_N{num_faults_to_inject}{batch_str}.txt"

    batch_size = 64
    top_faults = []
    total_failure_rate = 0.0
    PERC_SAVE_INTERVAL = 10
    save_points = set([int(faults_to_inject * x / 100) for x in range(PERC_SAVE_INTERVAL, 101, PERC_SAVE_INTERVAL)])
    pbar = tqdm(total=faults_to_inject, desc=f"Sequential FI N={num_faults_to_inject}{batch_str}")

    for inj_id, combo in enumerate(fault_combinations):
        # combo può essere lista di indici (large N, from pkl) oppure tuple (small N, random)
        if isinstance(combo[0], int):
            fault_tuple_list = [all_faults[i] for i in combo]
        else:
            fault_tuple_list = combo

        faults = [WeightFault(injection=inj_id, layer_name=layer, tensor_index=idx, bits=[bit], value=1)
                  for (layer, idx, bit) in fault_tuple_list]
        with torch.no_grad():
            injector = WeightFaultInjector(model)
            injector.inject_faults(faults, fault_mode='bit-flip')
            faulty_predictions = []
            for data, _ in test_loader:
                data = data.to(device)
                outputs = model(data)
                preds = torch.argmax(outputs, dim=1)
                faulty_predictions.append(preds.cpu())
            injector.restore_golden()
        faulty_predictions = torch.cat(faulty_predictions)
        total = len(clean_predictions)
        masked = (faulty_predictions == clean_predictions).sum().item()
        critical = total - masked
        accuracy = masked / total
        failure_rate = critical / total
        total_failure_rate += failure_rate
        if len(top_faults) < 100:
            heapq.heappush(top_faults, (failure_rate, inj_id, faults, accuracy))
        else:
            heapq.heappushpop(top_faults, (failure_rate, inj_id, faults, accuracy))

        if inj_id in save_points:
            perc = int(100 * inj_id / faults_to_inject)
            with open(output_path, 'a') as f:
                f.write(f"\n---- Progress: {perc}% ({inj_id} / {faults_to_inject}) ----\n")
                f.write(f"Top 100 most critical fault injections (NUM_FAULTS_TO_INJECT = {num_faults_to_inject})\n")
                f.write(f"Partial average failure rate: {total_failure_rate / (inj_id+1):.4f}\n\n")
                for rank, (failure_rate, inj_id_, faults, accuracy) in enumerate(sorted(top_faults, reverse=True), start=1):
                    fault_desc = ", ".join([f"{fault.layer_name}[{fault.tensor_index}] bit {fault.bits[0]}" for fault in faults])
                    f.write(f"#{rank:3d} | Injection {inj_id_:6d} | FR={failure_rate:.4f} | Acc={accuracy:.4f} | {fault_desc}\n")
            print(f"\n[INFO] Salvato aggiornamento top100 al {perc}% in {output_path}")

        pbar.update(1)
    pbar.close()

    actual_injections = inj_id + 1
    avg_failure_rate = total_failure_rate / actual_injections

    if actual_injections < faults_to_inject:
        print(f"[WARNING] Solo {actual_injections} combinazioni uniche generate su {faults_to_inject} richieste.")

    with open(output_path, 'a') as f:
        f.write(f"\n---- Progress: 100% ({actual_injections} / {faults_to_inject}) ----\n")
        f.write(f"Top 100 most critical fault injections (NUM_FAULTS_TO_INJECT = {num_faults_to_inject})\n")
        f.write(f"Final average failure rate: {avg_failure_rate:.4f} (based on {actual_injections} injections)\n\n")
        for rank, (failure_rate, inj_id_, faults, accuracy) in enumerate(sorted(top_faults, reverse=True), start=1):
            fault_desc = ", ".join([f"{fault.layer_name}[{fault.tensor_index}] bit {fault.bits[0]}" for fault in faults])
            f.write(f"#{rank:3d} | Injection {inj_id_:6d} | FR={failure_rate:.4f} | Acc={accuracy:.4f} | {fault_desc}\n")

    print(f"\nSaved top 100 critical faults (with progress) to {output_path}")

    with open(output_path_final, 'w') as f:
        f.write(f"Top 100 most critical fault injections (NUM_FAULTS_TO_INJECT = {num_faults_to_inject})\n")
        f.write(f"Average failure rate: {avg_failure_rate:.4f} (based on {actual_injections} injections)\n\n")
        for rank, (failure_rate, inj_id_, faults, accuracy) in enumerate(sorted(top_faults, reverse=True), start=1):
            fault_desc = ", ".join([f"{fault.layer_name}[{fault.tensor_index}] bit {fault.bits[0]}" for fault in faults])
            f.write(f"#{rank:3d} | Injection {inj_id_:6d} | FR={failure_rate:.4f} | Acc={accuracy:.4f} | {fault_desc}\n")

    print(f"\nSaved top 100 critical faults to {output_path_final}")
    duration = time.time() - start_time
    print(f"\nAnalysis completed in {duration/60:.2f} minutes.")
