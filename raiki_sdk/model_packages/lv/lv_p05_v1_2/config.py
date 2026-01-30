"""
Configuration for Louis Vuitton P05 V1.2 Model
"""
from raiki_sdk.utils.hf_lib import download_file_from_hf
# Model version
AUTH_MODEL_VERSION = "v1_2" 
YOLO_MODEL_VERSION = "v1"


# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/lv_p05_detector_v1",
    filename="lv_p05_detector_v1_yolo_240507_240508.pt"
)

AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/lv_p05_auth_v1_2",
    filename="lv_p05_auth_v1.2_maxvit_small_512_20260129_20260129.pth"
)

# AUTH parameters
TARGET_SIZE = 512
AUTH_MODEL_NAME = "maxvit_small_tf_512.in1k"
AUTH_NUM_CLASSES = 2
AUTH_CLASS_LABELS = ["fake_metal", "real_metal"]

# Metadata
CATEGORY = "bag-lv"
PART = "P05"

# Package name
PACKAGE_NAME = "lv_p05_v1_2"

# YOLO parameters
YOLO_CONF = 0.65
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (127, 127, 127)