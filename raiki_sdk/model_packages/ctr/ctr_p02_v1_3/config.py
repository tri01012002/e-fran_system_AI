"""
Configuration for CTR P02 Models Version 1_2
"""

from raiki_sdk.utils import download_file_from_hf
import cv2

# Model version
AUTH_MODEL_VERSION = "v1_3"
YOLO_MODEL_VERSION = "v1"


YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/ctr_p02_detector_v1-3",
    filename="ctr_p02_detector_v1-3_yolov26_260422_260423.pt"
)
AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/ctr_p02_auth_v1-3",
    filename="ctr_p02_auth_v1-3_dinov3_260421_260422.pth"
)
        
# AUTH parameters
TARGET_SIZE = 512
AUTH_MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"
AUTH_NUM_CLASSES = 2
AUTH_CLASS_LABELS = ['fake', 'real']

# Metadata
CATEGORY = "ctr"
PART = "P02"

# Package name
PACKAGE_NAME = "ctr_p02_v1_3"

# YOLO parameters
YOLO_CONF = 0.25
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (255, 255, 255)
