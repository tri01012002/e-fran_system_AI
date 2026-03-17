"""
Configuration for Omega P02 Model Version 1.4
"""
from raiki_sdk.utils import download_file_from_hf

# Model version
AUTH_MODEL_VERSION = "v1.4"
YOLO_MODEL_VERSION = "v1"

# convnextv2_base 384
AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/omg_p02_auth_v1-4",
    filename="omg_p02_auth_v1.4_convnextv2-base-384_best_model.pth"
)
NEEDLE_ALIGNER_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/omg_p02_needle_aligner_v1",
    filename="omg_p02_needle_aligner_v1_yolo11m_20251225_20251226.pt"
)
# Detector weights
DETECTOR_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/omg_p02_detector_v1",
    filename="omg_p02_detector_v1_rfdetr_20251125_20251125.pth"
)

# AUTH parameters
TARGET_SIZE = 384
AUTH_MODEL_NAME = "convnextv2_base.fcmae_ft_in22k_in1k_384"
AUTH_NUM_CLASSES = 2
AUTH_CLASS_LABELS = ["real", "fake"]

# Metadata
CATEGORY = "omega"
PART = "P02"

# Package name
PACKAGE_NAME = "omg_p02_v1_4"

# YOLO parameters
YOLO_NEEDLE_ALIGNER_CONF = 0.3
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (255, 255, 255)
