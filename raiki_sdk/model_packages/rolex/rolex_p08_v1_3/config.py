"""
Configuration for Rolex P08 Model
"""
from raiki_sdk.utils import download_file_from_hf
# Model version
AUTH_MODEL_VERSION = "v1_3" 
YOLO_MODEL_VERSION = "v1_3"


# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex_p08_detector_v1-3",
    filename="rolex_p08_detector_v1-3_yolo_241123_241124.pt"
)

AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex_p08_auth_v1-3",
    filename="rolex_p08_auth_v1-3_caformer_250505_250506.pth"
)

# AUTH parameters
TARGET_SIZE = 224
AUTH_MODEL_NAME = "caformer_m36.sail_in22k_ft_in1k"
AUTH_NUM_CLASSES = 2
AUTH_CLASS_LABELS = ['fake', 'real']

# Metadata
CATEGORY = "rolex"
PART = "P08"

# Package name
PACKAGE_NAME = "rolex_p08_v1_3"

# YOLO parameters
YOLO_CONF = 0.7
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (85, 85, 85)
