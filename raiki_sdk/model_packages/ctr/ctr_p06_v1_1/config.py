"""
Configuration for Omega P04 Models Version 1_1
"""
from raiki_sdk.utils import download_file_from_hf
import cv2

# Model version
AUTH_MODEL_VERSION = "v1_1"
YOLO_MODEL_VERSION = "v1"

# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/ctr_p06_detector_v1-1",
    filename="ctr_p06_detector_v1-1_20260206_20260206.pt"
)
AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/ctr_p06_auth_v1-1",
    filename="ctr_p06_auth_v1-1_onestage_maxvitbase_20260212_20260212.pth"
)

# AUTH parameters
TARGET_SIZE = 256
AUTH_MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"
AUTH_NUM_CLASSES = 2
AUTH_CLASS_LABELS = [
    "Fake_type1_level1",
    "Real_type1_level1"
]

# Metadata
CATEGORY = "ctr"
PART = "P06"

# Package name
PACKAGE_NAME = "ctr_p06_v1_1"

# YOLO parameters
YOLO_CONF = 0.65 
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (255, 255, 255)
