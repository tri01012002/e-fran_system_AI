"""
Configuration for Louis Vuitton P07 Models Version 1.6
"""
from raiki_sdk.utils import download_file_from_hf

# Model version
AUTH_MODEL_VERSION = "v1_6"
YOLO_MODEL_VERSION = "v1_3"

# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/lv_p02_detector_v1-3",
    filename="lv_p02_detector_v1-3_yolov11_250403_250404.pt"
)

AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/lv_p07_auth_v1_6",
    filename="lv_p07_auth_v1_6_cmal_210426_210426.pth"
)

# AUTH parameters
TARGET_SIZE = 384
AUTH_MODEL_NAME = "CMAL"
AUTH_NUM_CLASSES = 2
AUTH_CLASS_LABELS = ["fake", "real"]


# Metadata
CATEGORY = "bag-lv"
PART = "P07"

# Package name
PACKAGE_NAME = "lv_p07_v1_6"

# YOLO parameters
YOLO_CONF = 0.55
YOLO_IMGSZ = 640