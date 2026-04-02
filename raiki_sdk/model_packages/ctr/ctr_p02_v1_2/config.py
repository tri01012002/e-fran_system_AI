"""
Configuration for CTR P02 Models Version 1_2
"""

from raiki_sdk.utils import download_file_from_hf
import cv2

# Model version
AUTH_MODEL_VERSION = "v1_2"
YOLO_MODEL_VERSION = "v1_2"

# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/ctr_p02_detector_v1-2",
    filename="ctr_p02_detector_v1-2_yolov11s_260317_260323.pt"
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
PACKAGE_NAME = "ctr_p02_v1_2"

# YOLO parameters
YOLO_CONF = 0.35
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (255, 255, 255)
