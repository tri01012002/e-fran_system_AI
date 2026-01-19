"""
Configuration for Omega P01 Models Version 1_1
"""
from raiki_sdk.utils import download_file_from_hf
import cv2

# Model version
AUTH_MODEL_VERSION = "v1_1"
YOLO_MODEL_VERSION = "v1"

# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/omg_p01_detector_v1",
    filename="omg_p01_detector_v1_yolo11_20251114_20251117.pt"
)
AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/omg_p01_auth_v1-1",
    filename="omg_p01_auth_v1.1_maxvit_small_512_20251231_20260119.pth"
)

# AUTH parameters
TARGET_SIZE = 512
AUTH_MODEL_NAME = "maxvit_small_tf_512.in1k"
AUTH_NUM_CLASSES = 2
AUTH_CLASS_LABELS = ["fake", "real"]

# Metadata
CATEGORY = "omg"
PART = "P01"

# Package name
PACKAGE_NAME = "omg_p01_v1_1"

# YOLO parameters
YOLO_CONF = 0.65 
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (255, 255, 255)
