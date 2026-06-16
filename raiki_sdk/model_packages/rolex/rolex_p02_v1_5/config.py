"""
Configuration for Rolex P02 Models Version 1.5
"""
from raiki_sdk.utils import download_file_from_hf

# Model version
AUTH_MODEL_VERSION = "v1_5" 
YOLO_MODEL_VERSION = "v1_4"

# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex_p02_detector_v1-4", 
    filename="rolex_p02_detector_v1-4_yolov11_260415_260415.pt"
)

AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex_p02_auth_v1-5", 
    filename="rolex_p02_auth_v1-5_maxvitsmall_260415_260417.pth"
)

# AUTH parameters
TARGET_SIZE = 512
AUTH_MODEL_NAME = "maxvit_small_tf_512.in1k"  
AUTH_NUM_CLASSES = 4
AUTH_CLASS_LABELS = ["fake_crown", "fake_rolex", "real_crown", "real_rolex"]


# Metadata
CATEGORY = "rolex"
PART = "P02"

# Package name
PACKAGE_NAME = "rolex_p02_v1_5"

# YOLO parameters
YOLO_CONF = 0.6
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (127, 127, 127)