"""
Configuration for Louis Vuitton P02 Models Version 1.5
"""
from raiki_sdk.utils import download_file_from_hf

# Model version
AUTH_MODEL_VERSION = "v1_5"
YOLO_MODEL_VERSION = "v1_3"

# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/lv_p02_detector_v1-3",
    filename="lv_p02_detector_v1-3_yolov11_250403_250404.pt"
)

AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/lv_p02_auth_v1-5",
    filename="lv_p02_auth_v1-5_cmal_250907_251017.pth"
)

# AUTH parameters
TARGET_SIZE = 421
AUTH_MODEL_NAME = "CMAL"
AUTH_NUM_CLASSES = 8
AUTH_CLASS_LABELS = ["convex", "convex_fake", "flat_rivet", "flat_rivet_fake", "socket", "socket_fake", "stud", "stud_fake"]


# Metadata
CATEGORY = "bag-lv"
PART = "P02"

# Package name
PACKAGE_NAME = "lv_p02_v1_5"

# YOLO parameters
YOLO_CONF = 0.55
YOLO_IMGSZ = 640