import numpy as np
from utils import _analyze_fault_range_chunk
import os

# Inizializza variabile globale visibile da utils.py
import builtins
builtins.CLEAN_OUTPUT = np.load("output/clean_output/Banknote/SimpleMLP/batch_64/clean_output.npy", allow_pickle=True)

# Parametri per test
chunk_id = 0
fault_ids = list(range(5))
batch_folder = "output/faulty_output/Banknote/SimpleMLP/batch_64/bit-flip"
batch_size = 64
n_batches = 7
output_dir = "results_summary/Banknote/SimpleMLP/batch_64"

# Esegui test
_analyze_fault_range_chunk(chunk_id, fault_ids, batch_folder, batch_size, n_batches, output_dir)
