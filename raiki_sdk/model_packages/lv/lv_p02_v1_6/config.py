"""
Configuration for Louis Vuitton P02 Models Version 1.6
"""
from raiki_sdk.utils import download_file_from_hf

# Model version
AUTH_MODEL_VERSION = "v1_6"
YOLO_MODEL_VERSION = "v1_4"

# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/lv_p02_detector_v1-4",
    filename="lv_p02_detector_v1-4_yolov11_260326_260327.pt"
)

AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/lv_p02_auth_v1-6",
    filename="lv_p02_auth_v1-6_maxvit_260326_260327.pth"
)

# AUTH parameters
TARGET_SIZE = 512
AUTH_MODEL_NAME = "maxvit_small_tf_512.in1k"
AUTH_NUM_CLASSES = 2
AUTH_CLASS_LABELS = ["fake", "real"]


# Metadata
CATEGORY = "bag-lv"
PART = "P02"

# Package name
PACKAGE_NAME = "lv_p02_v1_6"

# YOLO parameters
YOLO_CONF = 0.55
YOLO_IMGSZ = 640