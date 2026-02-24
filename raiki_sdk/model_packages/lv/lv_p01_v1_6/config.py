"""
Configuration for Louis Vuitton P01 Models Version 1.6
"""
from raiki_sdk.utils import download_file_from_hf

# Model version
AUTH_MODEL_VERSION = "v1_6"
YOLO_MODEL_VERSION = "v1_3"

# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/lv_p01_detector_v1-3",
    filename="lv_p01_detector_yolov11_250624_250625.pt"
)

AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/lv_p01_auth_v1-6",
    filename="lv_p01_auth_v1-6_cmal_251230_251230.pth"
)

# AUTH parameters
TARGET_SIZE = 421
AUTH_MODEL_NAME = "CMAL"
AUTH_NUM_CLASSES = 2
AUTH_CLASS_LABELS = ["fake", "real"]


# Metadata
CATEGORY = "bag-lv"
PART = "P01"

# Package name
PACKAGE_NAME = "lv_p01_v1_6"

# YOLO parameters
YOLO_CONF = 0.55
YOLO_IMGSZ = 640
AREA_RATIO_VALID = 0.25