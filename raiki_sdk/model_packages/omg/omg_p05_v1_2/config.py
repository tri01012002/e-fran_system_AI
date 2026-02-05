"""
Configuration for Omega P05 Models Version 1.2
"""
from raiki_sdk.utils import download_file_from_hf
import cv2

# Model version
AUTH_MODEL_VERSION = "v1-2"
YOLO_MODEL_VERSION = "v1-3"

# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/omg_p05_detector_v1-3",
    filename="omg_p05_detector_v1-3_20260205_20260205.pt"
)
AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/omg_p05_auth_v1-2",
    filename="omg_p05_auth_v1.2_convnextv2_384_20260122_20260122.pth"
)
# AUTH parameters
TARGET_SIZE = 384
AUTH_MODEL_NAME = "convnextv2_base.fcmae_ft_in22k_in1k_384"
AUTH_NUM_CLASSES = 2
AUTH_CLASS_LABELS = ["fake", "real"]

# Metadata
CATEGORY = "omg"
PART = "P05"

# Package name
PACKAGE_NAME = "omg_p05_v1_2"

# YOLO parameters
YOLO_CONF = 0.65 
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (255, 255, 255)
ROTATION_CONFIG = {
    1: {"name": "down", "rotate": cv2.ROTATE_180},
    2: {"name": "left", "rotate": cv2.ROTATE_90_CLOCKWISE},
    3: {"name": "right", "rotate": cv2.ROTATE_90_COUNTERCLOCKWISE },
}