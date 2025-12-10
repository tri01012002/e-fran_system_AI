"""
Configuration for Omega P06 v1 Model
"""
from raiki_sdk.utils.hf_lib import download_file_from_hf
import cv2
# Model version
AUTH_MODEL_VERSION = "v1.1" 
YOLO_MODEL_VERSION = "v1"


# # Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/omg_p06_detector_v1",
    filename="omg_p06_detector_v1_yolo11_20251114.pt"
)

AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/omg_p06_auth_v1-1",
    filename="omg_p06_auth_v1.1_maxvit_small_512_20251210_20251210.pth"
)

# AUTH parameters
TARGET_SIZE = 512
AUTH_MODEL_NAME = "maxvit_small_tf_512.in1k"
AUTH_NUM_CLASSES = 2
AUTH_CLASS_LABELS = ["fake", "real"]

# Metadata
CATEGORY = "omg"
PART = "P06"

# Package name
PACKAGE_NAME = "omg_p06_v1"

# YOLO parameters
YOLO_CONF = 0.65
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (127, 127, 127)
# YOLO class names for P06 detector
YOLO_CLASS_NAMES = {
    0: 'back-cover-bottom',
    1: 'back-cover-left', 
    2: 'back-cover-right',
    3: 'back-cover-top'
}

# Rotation config to normalize detected images to bottom orientation
ROTATION_CONFIG = {
    0: {"name": "bottom", "rotate": None},
    1: {"name": "left", "rotate": cv2.ROTATE_90_COUNTERCLOCKWISE},
    2: {"name": "right", "rotate": cv2.ROTATE_90_CLOCKWISE},
    3: {"name": "top", "rotate": cv2.ROTATE_180},
}