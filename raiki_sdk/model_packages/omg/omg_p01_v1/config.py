"""
Configuration for Omega P01 Models Version 1
"""
from raiki_sdk.utils import download_file_from_hf
import cv2

# Model version
AUTH_MODEL_VERSION = "v1"
YOLO_MODEL_VERSION = "v1"

# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/omg_p01_detector_v1",
    filename="omg_p01_detector_v1_yolo11_20251114_20251117.pt"
)
AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/omg_p01_auth_v1-1_CMAL",
    filename="omg_p01_auth_v1-1_CMAL_20251125_20251125.pth"
)

# AUTH parameters
TARGET_SIZE = 368
AUTH_MODEL_NAME = "CMAL"
AUTH_NUM_CLASSES = 2
AUTH_CLASS_LABELS = ["fake", "real"]

# Metadata
CATEGORY = "omg"
PART = "P01"

# Package name
PACKAGE_NAME = "omg_p01_v1"

# YOLO parameters
YOLO_CONF = 0.65 
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (255, 255, 255)
