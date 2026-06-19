"""
Configuration for Louis Vuitton P04 Case 4 Models Version 1.6
"""
from raiki_sdk.utils import download_file_from_hf

# Model version
AUTH_MODEL_VERSION = "v1_6"
YOLO_MODEL_VERSION = "v1_3"

# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/lv_p04case4_detector_v1-3",
    filename="lv_p04case4_detector_v1-3_yolov11_19062026_19062026.pt"
)

AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/lv_p04case4_auth_v1-6",
    filename="lv_p04case4_auth_v1_6_maxvitsmall_260531_260531.pth"
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
PACKAGE_NAME = "lv_p04case4_v1_6"

# YOLO parameters
YOLO_CONF = 0.65
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (127, 127, 127)