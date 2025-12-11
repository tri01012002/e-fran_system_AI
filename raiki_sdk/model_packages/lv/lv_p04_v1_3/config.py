"""
Configuration for Louis Vuitton P04 Models Version 1.3
"""
from raiki_sdk.utils import download_file_from_hf

# Model version
AUTH_MODEL_VERSION = "v1_3"
YOLO_MODEL_1_VERSION = "v1_2"
YOLO_MODEL_2_VERSION = "v1_3"

# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS_1 = download_file_from_hf(
    repo_id="rikai-ai/lv_p04_detector_v1-2",
    filename="lv_p04_detector_v1-2_yolov11_250714_250715.pt" # Yolo detect
)

YOLO_WEIGHTS_2 = download_file_from_hf(
    repo_id="rikai-ai/lv_p04_detector_v1-3",
    filename="lv_p04_detector_v1-3_yolov11_250513_250514.pt" # Yolo crop
)

AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/lv_p04_auth_v1-3",
    filename="lv_p04_auth_v1-3_cmal_20251127_20251127.pth"
)

# AUTH parameters
TARGET_SIZE = 368
AUTH_MODEL_NAME = "CMAL"
AUTH_NUM_CLASSES = 4
AUTH_CLASS_LABELS = ['fake_circle','fake_itt','real_circle' ,'real_itt']


# Metadata
CATEGORY = "bag-lv"
PART = "P04"

# Package name
PACKAGE_NAME = "lv_p04_v1_3"

# YOLO parameters
YOLO_CONF = 0.40
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (255,255,255)
RESIZE_OUTPUT_SIZE = 368