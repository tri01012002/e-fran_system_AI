"""
Configuration for Rolex P09 Model
"""
from raiki_sdk.utils import download_file_from_hf
# Model version
AUTH_MODEL_VERSION = "v1" 
YOLO_MODEL_VERSION = "v1"


# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex_p09_detector_v1",
    filename="rolex_p09_detector_v1_yolo_241216_241217.pt"
)

AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex_p09_auth_v1",
    filename="rolex_p09_auth_v1_resnet50_241224_241224.pth"
)

# AUTH parameters
TARGET_SIZE = 224
AUTH_MODEL_NAME = "ResNet50"
AUTH_NUM_CLASSES = 2
AUTH_CLASS_LABELS = ['fake', 'real']

# Metadata
CATEGORY = "rolex"
PART = "P09"

# Package name
PACKAGE_NAME = "rolex_P09_v1_3"

# YOLO parameters
YOLO_CONF = 0.55
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (85, 85, 85)