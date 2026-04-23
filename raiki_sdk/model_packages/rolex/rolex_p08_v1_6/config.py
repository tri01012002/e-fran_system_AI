"""
Configuration for Rolex P08 v1.6 Model
"""
from raiki_sdk.utils import download_file_from_hf
# Model version
AUTH_MODEL_VERSION = "v1_6"
YOLO_MODEL_VERSION = "v1_4"

# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex_p08_detector_v1-4",
    filename="rolex_p08_detector_v1-4_yolov11_260421_260421.pt"
)

AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex_p08_auth_v1-6",
    filename="rolex_p08_auth_v1-6_maxvitsmall_260420_260422.pth"
)

# AUTH parameters
TARGET_SIZE = 512
AUTH_MODEL_NAME = "maxvit_small_tf_512.in1k"
AUTH_NUM_CLASSES = 2
AUTH_CLASS_LABELS = ['fake','real']


# Metadata
CATEGORY = "rolex"
PART = "P08"

# Package name
PACKAGE_NAME = "rolex_p08_v1_6"

# YOLO parameters
YOLO_CONF = 0.55
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (255, 255, 255)
