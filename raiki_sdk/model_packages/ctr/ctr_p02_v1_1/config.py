"""
Configuration for CTR P02 Models Version 1_1
"""

from raiki_sdk.utils import download_file_from_hf
import cv2

# Model version
AUTH_MODEL_VERSION = "v1_1"
YOLO_MODEL_VERSION = "v1"

# Model weights paths (relative to package root or absolute)

# USE OMG P02 DETECTOR
DETECTOR_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/omg_p02_detector_v1",
    filename="omg_p02_detector_v1_rfdetr_20251125_20251125.pth"
)
AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/ctr_p02_auth_v1-1",
    filename="ctr_p02_auth_v1-1_maxvitbase_20260212_20260212.pth"
)

# AUTH parameters
TARGET_SIZE = 256
AUTH_MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"
AUTH_NUM_CLASSES = 5
AUTH_CLASS_LABELS = ['Fake_type1_level1', 'Fake_type1_level2', 'Fake_type2_level2', 'Real_type1_level1', 'Real_type2_level1']

# Metadata
CATEGORY = "ctr"
PART = "P02"

# Package name
PACKAGE_NAME = "ctr_p02_v1_1"

# YOLO parameters
YOLO_CONF = 0.65 
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (255, 255, 255)
