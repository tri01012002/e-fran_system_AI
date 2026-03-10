"""
Configuration for Rolex P05 Model
"""
from raiki_sdk.utils import download_file_from_hf
# Model version
AUTH_MODEL_VERSION = "v1_4" 
YOLO_MODEL_VERSION = "v1_3"


# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex_p05_detector_v1-3",
    filename="rolex_p05_detector_v1-3_yolo_241202_241203.pt"
)

AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex_p05_auth_v1-5",
    filename="rolex_p05_auth_v1-5_maxvitsmall-512_260304_260304.pth"
)

# AUTH parameters
TARGET_SIZE = 512
AUTH_MODEL_NAME = "maxvit_small_tf_512.in1k"
AUTH_NUM_CLASSES = 4
AUTH_CLASS_LABELS = ['fake', 'real', 'old', "new"]

# Metadata
CATEGORY = "rolex"
PART = "P05"

# Package name
PACKAGE_NAME = "rolex_p05_v1_4"

# YOLO parameters
YOLO_CONF = 0.25
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (127, 127, 127)