"""
Configuration for Louis Vuitton P04 Case 4 Models Version 1.6
"""
from raiki_sdk.utils import download_file_from_hf

# Model version
AUTH_MODEL_VERSION = "v1_6"
YOLO_MODEL_VERSION = "v1_2"

# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/lv_p04case4_detector_v1-2",
    filename="lv_p04case4_detector_v1-2_yolov11_250714_250715.pt"
)

AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/lv_p04_auth_v1-6",
    filename="lv_p04_auth_v1_6_maxvitsmall.pth"
)

# AUTH parameters
TARGET_SIZE = 512
AUTH_MODEL_NAME = "maxvit_small_tf_512.in1k"
AUTH_NUM_CLASSES = 2
AUTH_CLASS_LABELS = ['fake','real']

# Training hyperparameters (from retrain notebook)
EPOCHS          = 60
BATCH_SIZE      = 16
LR_HEAD         = 1e-3
LR_BACKBONE     = 1e-5
WEIGHT_DECAY    = 1e-4
LABEL_SMOOTHING = 0.1
FOCAL_GAMMA     = 2.0
WARMUP_EPOCHS   = 3
UNFREEZE_EPOCH  = 5
PATIENCE        = 10
ES_DELTA        = 1e-4
MONITOR_METRIC  = 'val_f1_macro'

# Data split ratios
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
RANDOM_SEED  = 42
MIN_IMG_SIZE = 100

# Metadata
CATEGORY = "bag-lv"
PART = "LV_P04case4"

# Package name
PACKAGE_NAME = "lv_p04case4_v1_6"

# YOLO parameters
YOLO_CONF = 0.35
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (127, 127, 127)