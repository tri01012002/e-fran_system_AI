"""
Configuration for Louis Vuitton P04 Case 4 Models Version 1.4
"""
from raiki_sdk.utils import download_file_from_hf

# Model version
AUTH_MODEL_VERSION = "v1_4"
YOLO_MODEL_VERSION = "v1_2"

# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/lv_p04case4_detector_v1-2",
    filename="lv_p04case4_detector_v1-2_yolov11_250714_250715.pt"
)

AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/lv_p04case4_auth_v1-4",
    filename="lv_p04case4_auth_v1.4_convnext-small_20251204_20251204.pth"
)

# AUTH parameters
TARGET_SIZE = 224
AUTH_MODEL_NAME = "convnext_small.in12k"
AUTH_NUM_CLASSES = 2
AUTH_CLASS_LABELS = ['fake','real']


# Metadata
CATEGORY = "bag-lv"
PART = "LV_P04case4"

# Package name
PACKAGE_NAME = "lv_p04case4_v1_4"

# YOLO parameters
YOLO_CONF = 0.35
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (255, 255, 255)