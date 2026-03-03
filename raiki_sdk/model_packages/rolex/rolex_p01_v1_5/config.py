"""
Configuration for Rolex P01 Model
"""
from raiki_sdk.utils import download_file_from_hf
# Model version
AUTH_MODEL_VERSION = "v1_5" 
YOLO_MODEL_VERSION = "v1_3"


# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex_p01_detector_v1-3", 
    filename="rolex_p01_detector_v1-3_yolov11m_241226_241226.pt"
)

AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex_p01_auth_v1-5", 
    filename="rolex_p01_auth_v1-5_maxvitsmall512_260226_260227.pth"
)

# AUTH parameters
TARGET_SIZE = 512
AUTH_MODEL_NAME = "maxvit_small_tf_512.in1k"
AUTH_NUM_CLASSES = 6
AUTH_CLASS_LABELS = ['fake_crown', 'fake_marker', 'fake_rolex','real_crown', 'real_marker', 'real_rolex']
COLOR_PADDING = (127, 127, 127)
# Metadata
CATEGORY = "rolex"
PART = "P01"

# Package name
PACKAGE_NAME = "rolex_p01_v1_5"

# YOLO parameters
YOLO_CONF = 0.55
YOLO_IMGSZ = 640