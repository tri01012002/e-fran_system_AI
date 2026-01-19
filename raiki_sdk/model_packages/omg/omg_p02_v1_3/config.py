"""
Configuration for Omega P02 Model Version 1.3
"""
from raiki_sdk.utils import download_file_from_hf
# Model version
AUTH_MODEL_VERSION = "v1.3" 
YOLO_MODEL_VERSION = "v1"

# maxvit small 512
AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/omg_p02_auth_v1-3",
    filename="omg_p02_auth_v1.3_maxvit-small-512_20260116_20260116.pth"
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
TARGET_SIZE = 512
AUTH_MODEL_NAME = "maxvit_small_tf_512.in1k"
AUTH_NUM_CLASSES = 2
AUTH_CLASS_LABELS = ["fake", "real"]

# Metadata
CATEGORY = "omega"
PART = "P02"

# Package name
PACKAGE_NAME = "omg_p02_v1_3"

# YOLO parameters
YOLO_NEEDLE_ALIGNER_CONF = 0.3
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (255, 255, 255)