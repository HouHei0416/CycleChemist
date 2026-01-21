import os

"""
Central configuration for MOE2 (pretraining) and P3 (PCE training).

The constants are grouped into:
  1) SHARED MODEL ARGS         used by both pretrain.py and train_pce.py
  2) PRETRAIN ARGS (MOE2)      used only by pretrain.py
  3) PCE TRAINING ARGS (P3)    used only by train_pce.py
  4) DATA PATHS                shared file locations
  5) MODEL PATHS               where .pth files are stored

Names are kept the same so existing import sites continue to work.
"""

########################################
# 1) SHARED MODEL ARGS (MOE2 + P3)
########################################

IN_CHANNELS = 31          # Node feature dimension
EDGE_DIM = 15             # Edge feature dimension
HIDDEN_CHANNELS = 512
OUT_CHANNELS = 512

HOMOLUMO_TARGETS = 2      # Number of HOMO/LUMO regression targets
PCE_TARGETS = 1           # Number of PCE regression targets
REGRESSION_TARGETS = 2    # Used by MOE2 for HOMO/LUMO

HEADS = 8                 # Attention heads for GATv2
DROPOUT_RATE = 0.25
FINGERPRINT_DIM = 2048    # Morgan fingerprint dimension (if used downstream)

########################################
# 2) PRETRAIN ARGS (used in pretrain.py)
########################################

SEED = 42
VAL_RATIO = 0.15

# Pretraining epochs
MLM_EPOCHS = 50           # Masked node modeling epochs
HOMO_EPOCHS = 50          # HOMO/LUMO regression epochs

# Learning rates for pretraining stages
MLM_LEARNING_RATE = 1e-5
HOMO_LEARNING_RATE = 1e-4

# Early stopping shared with P3 (both scripts import it)
EARLY_STOPPING_PATIENCE = 30
MIN_DELTA = 0.0001        # Minimum improvement for early stopping

########################################
# 3) PCE TRAINING ARGS (used in train_pce.py)
########################################

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
TEST_RATIO = 0.1          # Currently unused, kept for compatibility

########################################
# 4) DATA PATHS (shared)
########################################

# Directory where this config file lives (property_predictors/moe2_p3)
_CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root: CycleChemist
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_CONFIG_DIR))

# Data directory: CycleChemist/data
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data")

# Pretraining data
MLM_DATA_PATH = os.path.join(_DATA_DIR, "mmc2.csv")
HOMO_DATA_PATH = os.path.join(_DATA_DIR, "mmc2.csv")
HOMO_EXP_DATA_PATH = os.path.join(_DATA_DIR, "exp_dataset.csv")

# PCE training / inference data
PREDICT_DATA_PATH = os.path.join(_DATA_DIR, "exp_dataset.csv")

########################################
# 5) Checkpoints PATHS (shared)
########################################

# Base checkpoint directory: property_predictors/moe2_p3/ckpt
_MODEL_DIR = os.path.join(_CONFIG_DIR, "ckpt")

# MOE2 models (Molecular Orbital Energy Estimator)
_MOE2_DIR = os.path.join(_MODEL_DIR, "moe2")
MLM_MODEL_PATH = os.path.join(_MOE2_DIR, "MLM_model.pth")
HOMOLUMO_MODEL_PATH = os.path.join(_MOE2_DIR, "homolumo_model.pth")
HOMOLUMO_EXP_MODEL_PATH = os.path.join(_MOE2_DIR, "homolumo_exp_model.pth")

# P3 models (Photovoltaic Performance Predictor)
_P3_DIR = os.path.join(_MODEL_DIR, "p3")
PCE_MODEL_PATH = _P3_DIR
