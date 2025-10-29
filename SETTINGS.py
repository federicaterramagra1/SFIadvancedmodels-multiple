import torch

# ==============================
# ESPERIMENTO: scegli dataset & rete
# ==============================
DATASET_NAME = 'Wine'      # es.:  'Banknote' | 'Wine' | 'DryBean' | 'CIFAR10'
NETWORK_NAME = 'WineMLP'   # es.: 'SimpleMLP' | 'WineMLP' | 'BeanMLP' | 'ResNet20'
NUM_EPOCHS = 200
USE_ADVANCED_TRAIN = False

# Alias usati dal codice
DATASET = DATASET_NAME
NETWORK = NETWORK_NAME

# ==============================
# TOGGLE DI ESECUZIONE
# ==============================
ONLY_CLEAN_INFERENCE   = False   # esegui solo inferenza "clean"
FAULT_LIST_GENERATION  = True  # genera la fault list
FAULTS_INJECTION       = True  # lancia campagna d'iniezione
FI_ANALYSIS            = True  # post-analisi (classi 0..4) -> output_analysis.csv
FI_ANALYSIS_SUMMARY    = True  # sintesi -> *_summary.csv + .txt

# ==============================
# PARAMETRI FAULT INJECTION
# ==============================
SEED = 42
NUM_FAULTS_TO_INJECT = 1       # numero di bit flip simultanei per iniezione
FAULTS_TO_INJECT     = -1       # -1 = esaustiva sulla fault list; >0 = campiona N iniezioni
FAULT_MODEL          = 'bit-flip'  # 'stuck-at_params' | 'bit-flip' 
STUCK_VALUE          = 1
THRESHOLD            = 0.0       # soglia per errori non rilevati (se usata a valle)

# ==============================
# DISPOSITIVO DI CALCOLO
# ==============================
USE_CUDA_0 = True
USE_CUDA_1 = False

# ==============================
# DATI / BATCH
# ==============================
BATCH_SIZE   = 64
DATASET_PATH = 'Datasets/'
MODELS_PATH  = 'dlModels/'

# ==============================
# PATH PRETRAINED (se servono)
# ==============================
MODEL_TH_PATH  = f'dlModels/{DATASET}/pretrained/{NETWORK}_{DATASET}.th'
MODEL_PT_PATH  = f'dlModels/{DATASET}/pretrained/{NETWORK}_{DATASET}.pt'
MODEL_PTH_PATH = f'dlModels/{DATASET}/pretrained/{NETWORK}_{DATASET}.pth'

# ==============================
# CARTELLE DI OUTPUT
# ==============================
CLEAN_FM_FOLDER    = f'output/clean_feature_maps/{DATASET}/{NETWORK}/batch_{BATCH_SIZE}'
CLEAN_OUTPUT_FOLDER= f'output/clean_output/{DATASET}/{NETWORK}/batch_{BATCH_SIZE}'
FAULTY_FM_FOLDER   = f'output/faulty_feature_maps/{DATASET}/{NETWORK}/batch_{BATCH_SIZE}/{FAULT_MODEL}'
FAULTY_OUTPUT_FOLDER = f'output/faulty_output/{DATASET}/{NETWORK}/batch_{BATCH_SIZE}'

# ==============================
# ANALISI (post-campagna)
# ==============================
FI_ANALYSIS_PATH       = f'results_summary/{DATASET}/{NETWORK}/batch_{BATCH_SIZE}/'
FI_SUM_ANALYSIS_PATH   = f'results_summary/{DATASET}/{NETWORK}/batch_{BATCH_SIZE}/{NETWORK}_summary.csv'
RAM_LIMIT   = False
BATCH_START = 0
BATCH_END   = 13

# ==============================
# SALVATAGGI
# ==============================
SAVE_CLEAN_OFM   = True
SAVE_FAULTY_OFM  = True
SAVE_FAULTY_OUTPUT = True
INPUT_FMAPS_TO_SAVE = 'fc1' if SAVE_FAULTY_OFM else None  # usa None se non ti serve

# ==============================
# FAULT LIST (file)
# ==============================
FAULT_LIST_PATH = f'output/fault_list/{DATASET_NAME}/{NETWORK_NAME}/'
FAULT_LIST_NAME = f'{NETWORK_NAME}_{SEED}_fault_list_N{NUM_FAULTS_TO_INJECT}.csv'

# ==============================
# CLASSI MODULO (per salvataggi / FI)
# ==============================
MODULE_CLASSES = torch.nn.Linear      # per OFM/IFM
MODULE_CLASSES_FAULT_LIST = (torch.nn.Linear, torch.nn.Conv2d)       # per lista iniettabile

# ==============================
# PARAMETRI STATISTICI (stima N esperimenti)
# ==============================
error_margin = 0.01
probability = 0.5
confidence_constant = 2.58
bit = 8

# ==============================
# OPZIONALE: carica modello già addestrato
# ==============================
LOAD_MODEL_FROM_PATH = False
LOAD_MODEL_PATH = f"trained_models/{DATASET}_{NETWORK}_trained.pth"

# Log
NO_LOG_RESULTS = False
