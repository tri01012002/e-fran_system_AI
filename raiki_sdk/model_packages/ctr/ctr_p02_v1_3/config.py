"""
Configuration for CTR P02 Models Version 1_3
"""

from raiki_sdk.utils import download_file_from_hf
import cv2

# Model version
AUTH_MODEL_VERSION = "v1_3"
YOLO_MODEL_VERSION = "v1"

DETECTOR_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/omg_p02_detector_v1",
    filename="omg_p02_detector_v1_rfdetr_20251125_20251125.pth"
)

AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/ctr_p02_auth_v1-2",
    filename="ctr_p02_auth_v1_2_maxvitsmall_260317_260325.pth"
)

# AUTH parameters
TARGET_SIZE = 512
AUTH_MODEL_NAME = "maxvit_small_tf_512.in1k"
AUTH_NUM_CLASSES = 2
AUTH_CLASS_LABELS = ['fake', 'real']

# Metadata
CATEGORY = "ctr"
PART = "P02"

# Package name
PACKAGE_NAME = "ctr_p02_v1_3"

# YOLO parameters
YOLO_CONF = 0.65
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (255, 255, 255)
