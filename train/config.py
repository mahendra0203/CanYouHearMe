import torch

#Dataset names
DATASET_NAME ="mahendra0203/musiccaps_processed_full"

# Model settings
WHISPER_MODEL_NAME = "openai/whisper-large-v2"
LLM_MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"

# Data settings
BATCH_SIZE = 8
NUM_WORKERS = 8

# Training settings
LEARNING_RATE = 1.5e-3
NUM_TRAIN_EPOCHS = 1
WARMUP_STEPS = 250
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 250
EVAL_STEPS = 500
SAVE_STEPS = 1000
MAX_STEPS = 3000

# Device settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Wandb settings
WANDB_PROJECT = "multilm-llama3-whisper-large-v2"
WANDB_RUN_NAME = "refactored_run"

# Paths
OUTPUT_DIR = "./results"
LOGGING_DIR = "./logs"