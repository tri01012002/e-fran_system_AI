"""
Configuration for Rolex P02 Models Version 1.4
"""
from raiki_sdk.utils import download_file_from_hf

# Model version
AUTH_MODEL_VERSION = "v1_4" 
YOLO_MODEL_VERSION = "v1_3"

# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex_p02_detector_v1-3", 
    filename="rolex_p02_detector_v1-3_yolov11m_241125_241220.pt"
)

AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex_p02_auth_v1-4", 
    filename="rolex_p02_auth_v1-4_cmal_250918_251016.pth"
)

# AUTH parameters
TARGET_SIZE = 421
AUTH_MODEL_NAME = "CMAL"  
AUTH_NUM_CLASSES = 5
AUTH_CLASS_LABELS = ["Fake", "GroupE", "Indented_LeftParttern", "Indented_RightParttern", "Raised_RightParttern"]


# Metadata
CATEGORY = "rolex"
PART = "P02"

# Package name
PACKAGE_NAME = "rolex_p02_v1_4"

# YOLO parameters
YOLO_CONF = 0.55
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (85, 85, 85)