"""
Configuration for Rolex P01 Model
"""
from raiki_sdk.utils import download_file_from_hf
# Model version
AUTH_MODEL_VERSION = "v1_4" 
YOLO_MODEL_VERSION = "v1_3"


# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex_p01_detector_v1-3", 
    filename="rolex_p01_detector_v1-3_yolov11m_241226_241226.pt"
)

AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex_p01_auth_v1-4", 
    filename="rolex_p01_auth_v1-4_efficientNetB4_241226_250926.pth"
)

# AUTH parameters
TARGET_SIZE = 224
AUTH_MODEL_NAME = "tf_efficientnet_b4_ns"
AUTH_NUM_CLASSES = 3
AUTH_CLASS_LABELS = ['fake', 'crow_rolex', 'crow_triangle_rolex']

# Metadata
CATEGORY = "rolex"
PART = "P01"

# Package name
PACKAGE_NAME = "rolex_p01_v1_3"

# YOLO parameters
YOLO_CONF = 0.55
YOLO_IMGSZ = 640