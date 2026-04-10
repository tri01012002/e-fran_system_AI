"""
Configuration for Rolex P01 Model
"""
from raiki_sdk.utils import download_file_from_hf
# Model version
AUTH_MODEL_VERSION = "v1_6" 
YOLO_MODEL_VERSION = "v1_4"


# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex_p01_detector_v1-4", 
    filename="rolex_p01_detector_v1-4_yolov11_260402_260406.pt"
)

AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex_p01_auth_v1-6", 
    filename="rolex_p01_auth_v1-6_maxvitsmall_260406_260410.pth"
)

# AUTH parameters
TARGET_SIZE = 512
AUTH_MODEL_NAME = "maxvit_small_tf_512.in1k"
AUTH_NUM_CLASSES = 6
AUTH_CLASS_LABELS = ['fake_crown', 'fake_marker', 'fake_rolex', 'real_crown', 'real_marker', 'real_rolex']
COLOR_PADDING = (127, 127, 127)

# Metadata
CATEGORY = "rolex"
PART = "P01"

# Package name
PACKAGE_NAME = "rolex_p01_v1_6"

# YOLO parameters
YOLO_CONF = 0.55
YOLO_IMGSZ = 640