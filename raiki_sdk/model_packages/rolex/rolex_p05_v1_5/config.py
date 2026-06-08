"""
Configuration for Rolex P05 Model
"""
from raiki_sdk.utils import download_file_from_hf
# Model version
AUTH_MODEL_VERSION = "v1_5" 
YOLO_MODEL_VERSION = "v1_3"


# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex_p05_detector_v1-3",
    filename="rolex_p05_detector_v1-3_yolo_241202_241203.pt"
)

AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex_p05_auth_v1_5",
    filename="rolex_p05_auth_v1_5_maxvit384_290526_080626.pth"
)

# AUTH parameters
TARGET_SIZE = 384
AUTH_MODEL_NAME = "maxvit_small_tf_384.in1k"
AUTH_NUM_CLASSES = 2
AUTH_CLASS_LABELS = ['fake', 'real']

# Metadata
CATEGORY = "rolex"
PART = "P05"

# Package name
PACKAGE_NAME = "rolex_p05_v1_5"

# YOLO parameters
YOLO_CONF = 0.25
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (127, 127, 127)