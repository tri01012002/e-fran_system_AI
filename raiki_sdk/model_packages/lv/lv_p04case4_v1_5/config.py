"""
Configuration for Louis Vuitton P04 Case 4 Models Version 1.4
"""
from raiki_sdk.utils import download_file_from_hf

# Model version
AUTH_MODEL_VERSION = "v1_5"
YOLO_MODEL_VERSION = "v1_2"

# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/lv_p04case4_detector_v1-2",
    filename="lv_p04case4_detector_v1-2_yolov11_250714_250715.pt"
)

AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/lv_p04_auth_v1-5",
    filename="lv_p04_auth_v1_5_maxvitsmall_260304_260308.pth"
)

# AUTH parameters
TARGET_SIZE = 512
AUTH_MODEL_NAME = "maxvit_small_tf_512.in1k"
AUTH_NUM_CLASSES = 2
AUTH_CLASS_LABELS = ['fake','real']


# Metadata
CATEGORY = "bag-lv"
PART = "LV_P04case4"

# Package name
PACKAGE_NAME = "lv_p04case4_v1_5"

# YOLO parameters
YOLO_CONF = 0.35
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (127, 127, 127)