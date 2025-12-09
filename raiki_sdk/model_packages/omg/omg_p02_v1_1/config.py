"""
Configuration for Omega P02 Model Version 1.1
"""
from raiki_sdk.utils import download_file_from_hf
# Model version
AUTH_MODEL_VERSION = "v1.1" 
YOLO_MODEL_VERSION = "v1"

# maxvit small 512
AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/omg_p02_auth_v1-1",
    filename="omg_p02_auth_v1.1_maxvit-small-512_20251208_20251209.pth"
)

# Detector weights
DETECTOR_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/omg_p02_detector_v1",
    filename="omg_p02_detector_v1_rfdetr_20251125_20251125.pth"
)

# AUTH parameters
TARGET_SIZE = 512
AUTH_MODEL_NAME = "maxvit_small_tf_512.in1k"
AUTH_NUM_CLASSES = 9
AUTH_CLASS_LABELS = ["type1_fake", "type1_old_real", "type1_real", "type1_white_fake", "type1_white_real", "type2_fake", "type2_real", "type3_fake", "type3_real"]

# Metadata
CATEGORY = "omega"
PART = "P02"

# Package name
PACKAGE_NAME = "omg_p02_v1_1"

# YOLO parameters
YOLO_CONF = 0.5
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (255, 255, 255)