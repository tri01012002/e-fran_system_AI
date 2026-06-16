"""
Configuration for Omega P01 Models Version 1_1
"""
from raiki_sdk.utils import download_file_from_hf
import cv2

# Model version
AUTH_MODEL_VERSION = "v1_2"
YOLO_MODEL_VERSION = "v1"

# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/ctr_p01_detector_v1-1",
    filename="ctr_p01_detector_v1-1_20260206_20260206.pt"
)
AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/ctr_p01_auth_v1-2",
    filename="ctr_p01_auth_v1-2_convnextv2_20260319_20260319.pth"
)

# AUTH parameters
TARGET_SIZE = 384
AUTH_MODEL_NAME = "convnextv2_nano.fcmae_ft_in22k_in1k_384"
AUTH_NUM_CLASSES = 2
AUTH_CLASS_LABELS = ['fake', 'real']

# Metadata
CATEGORY = "ctr"
PART = "P01"

# Package name
PACKAGE_NAME = "ctr_p01_v1_2"

# YOLO parameters
YOLO_CONF = 0.8 
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (255, 255, 255)
