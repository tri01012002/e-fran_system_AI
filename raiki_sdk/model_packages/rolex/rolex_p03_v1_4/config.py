"""
Configuration for Rolex P03 Model
"""
from raiki_sdk.utils.hf_lib import download_file_from_hf
# Model version
AUTH_MODEL_VERSION = "v1_4" 
YOLO_MODEL_VERSION = "v1_3"


# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex_p03_detector_v1-3",
    filename="rolex_p03_detector_v1-3_yolov11m_241129_251129.pt"
)

AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex_p03_auth_v1-4",
    filename="rolex_p03_auth_v1-4_efficientNet_250818_250820.pt"
)

# AUTH parameters
TARGET_SIZE = 256
AUTH_MODEL_NAME = "EfficientNet_V2_S_Weights.IMAGENET1K_V1"
AUTH_NUM_CLASSES = 3
AUTH_CLASS_LABELS = [
    'fake',
    'real_normal',
    'real_old'
]

# Metadata
CATEGORY = "rolex"
PART = "P03"

# Package name
PACKAGE_NAME = "rolex_p03_v1_4"

# YOLO parameters
YOLO_CONF = 0.25
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (85, 85, 85)
RESIZE_OUTPUT_SIZE = 256