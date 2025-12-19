"""
Configuration for Rolex P10 Model
"""
from raiki_sdk.utils import download_file_from_hf
# Model version
AUTH_MODEL_VERSION = "v1_1" 
YOLO_MODEL_VERSION = "v1"


# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex_p10_detector_v1",
    filename="rolex_p10_detector_v1_yolo_251005_251006.pt"
)

AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex_p10_auth_v1-1",
    filename="rolex_p10_auth_v1-1_maxvittiny_250909_250910.pth"
)

# AUTH parameters
TARGET_SIZE = 512
AUTH_MODEL_NAME = "maxvit_tiny_tf_512.in1k"
AUTH_NUM_CLASSES = 2
AUTH_CLASS_LABELS = ['fake', 'real']

# Metadata
CATEGORY = "rolex"
PART = "P10"

# Package name
PACKAGE_NAME = "rolex_P10_v1_3"

# YOLO parameters
YOLO_CONF = 0.65
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (85, 85, 85)