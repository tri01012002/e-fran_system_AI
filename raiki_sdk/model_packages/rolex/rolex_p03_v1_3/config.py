"""
Configuration for Rolex P01 Model
"""
from raiki_sdk.utils.hf_lib import download_file_from_hf
# Model version
AUTH_MODEL_VERSION = "v1_3" 
YOLO_MODEL_VERSION = "v1_3"


# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex_p03_detector_v1-3",
    filename="rolex_p03_detector_v1-3_yolov11m_241129_251129.pt"
)

AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex_p03_auth_v1-3",
    filename="rolex_p03_auth_v1-3_efficientNet_250818_250820.pt"
)

# AUTH parameters
TARGET_SIZE = 256
AUTH_MODEL_NAME = "tf_efficientnet_b4_ns"
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
PACKAGE_NAME = "rolex_p03_v1_3"

# YOLO parameters
YOLO_CONF = 0.25
YOLO_IMGSZ = 640