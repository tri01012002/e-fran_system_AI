"""
Configuration for Rolex P04 Model
"""
from raiki_sdk.utils import download_file_from_hf
# Model version
AUTH_MODEL_1_VERSION = "v1_3"
AUTH_MODEL_2_VERSION = "v1_4"
YOLO_MODEL_VERSION = "v1_3"


# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex_p04_detector_v1_3",
    filename="rolex_p04_detector_v1_3_yolov11m_241203_241204.pt"
)

AUTH_WEIGHTS_1 = download_file_from_hf(
    repo_id="rikai-ai/rolex_p04_auth_v1-3",
    filename="rolex_p04_auth_v1_3_ggnet_241109_241112.pth"
)

AUTH_WEIGHTS_2 = download_file_from_hf(
    repo_id="rikai-ai/rolex_p04_auth_v1-4",
    filename="rolex_p04_auth_v1-4_efficientnet_250703_250709.pt"
)

# AUTH parameters
TARGET_SIZE = 224
AUTH_MODEL_1_NAME = "GGnet"
AUTH_MODEL_2_NAME = "efficientnet_v2_s"
AUTH_MODEL_1_NUM_CLASSES = 2
AUTH_MODEL_1_CLASS_LABELS = [
    'fake',
    'real',
]
AUTH_MODEL_2_NUM_CLASSES = 3
AUTH_MODEL_2_CLASS_LABELS = [
    'broken', 'real', 'superfake'
]

# Metadata
CATEGORY = "rolex"
PART = "P04"

# Package name
PACKAGE_NAME = "rolex_p04_v1_3"

# YOLO parameters
YOLO_CONF = 0.35
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (85, 85, 85)