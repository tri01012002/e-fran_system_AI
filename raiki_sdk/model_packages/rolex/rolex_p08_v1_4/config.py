"""
Configuration for Rolex P08 Model
"""
from raiki_sdk.utils import download_file_from_hf
# Model version
AUTH_MODEL_VERSION = "v1_4" 
YOLO_MODEL_VERSION = "v1_3"
FILTER_MODEL_VERSION = "v1"


# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex_p08_detector_v1-3",
    filename="rolex_p08_detector_v1-3_yolo_241123_241124.pt"
)

AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex_p08_auth_v1-4",
    filename="rolex_p08_auth_v1.4_CMAL_251125_251126.pth"
)

FILTER_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex_p08_filter_v1",
    filename="rolex_p08_filter_v1_CMAL_251124_251125.pth"
)

# AUTH parameters
TARGET_SIZE = 421
AUTH_MODEL_NAME = "CMAL"
AUTH_NUM_CLASSES = 4
FILTER_NUM_CLASSES = 4

AUTH_CLASS_LABELS = [
    "original_fake",
    "original_real",
    "yellow_rolex_fake",
    "yellow_rolex_real",
]

FILTER_CLASS_LABELS = [
    "yellow_rolex",
    "four_dots",
    "no_dots",
    "three_dots",
]

# Metadata
CATEGORY = "rolex"
PART = "P08"

# Package name
PACKAGE_NAME = "rolex_p08_v1_4"

# YOLO parameters
YOLO_CONF = 0.55
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (85, 85, 85)
