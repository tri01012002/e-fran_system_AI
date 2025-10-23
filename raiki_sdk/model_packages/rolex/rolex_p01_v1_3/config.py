"""
Configuration for Rolex P01 Model
"""
from raiki_sdk.config import MODEL_DIR
from huggingface_hub import hf_hub_download

# Model version
AUTH_MODEL_VERSION = "v1_3" 
YOLO_MODEL_VERSION = "v1_3"


# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = hf_hub_download(
    repo_id="rikai-ai/rolex_p01_detector_v1-3", 
    filename="rolex_p01_detector_v1-3_yolov11m_241226_241226.pt", 
    cache_dir=MODEL_DIR)

AUTH_WEIGHTS = hf_hub_download(
    repo_id="rikai-ai/rolex_p01_auth_v1-3", 
    filename="rolex_p01_auth_v1-3_efficientNetB4_241226_250926.pth", 
    cache_dir=MODEL_DIR)

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