"""
Configuration for Louis Vuitton P06 Models Version 1.1
"""
from raiki_sdk.utils import download_file_from_hf

# Model version
AUTH_MODEL_VERSION = "v1_1"
YOLO_MODEL_VERSION = "v1_3"

# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/lv_p02_detector_v1-3",
    filename="lv_p02_detector_v1-3_yolov11_250403_250404.pt"
)

AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/lv_p06_auth_v1_1",
    filename="lv_p06_auth_v1.1_cmal_368_20260129_20260129.pth"
)

# AUTH parameters
TARGET_SIZE = 368
AUTH_MODEL_NAME = "CMAL"
AUTH_NUM_CLASSES = 2
AUTH_CLASS_LABELS = ["stud", "stud_fake"]


# Metadata
CATEGORY = "bag-lv"
PART = "P06"

# Package name
PACKAGE_NAME = "lv_p06_v1_1"

# YOLO parameters
YOLO_CONF = 0.55
YOLO_IMGSZ = 640