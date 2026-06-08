"""
Configuration for Louis Vuitton P07 V1.7 Model
"""
from raiki_sdk.utils.hf_lib import download_file_from_hf
# Model version
AUTH_MODEL_VERSION = "v1_7" 
YOLO_MODEL_VERSION = "v1_3"


# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/lv_p02_detector_v1-3",
    filename="lv_p02_detector_v1-3_yolov11_250403_250404.pt"
)


AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/lv_p07_auth_v1-7",
    filename="lv_p07_auth_v1-7_convnext_200526_210526.pth"
)

# AUTH parameters
TARGET_SIZE = 448
AUTH_MODEL_NAME = "convnext_base.fb_in22k_ft_in1k"
AUTH_NUM_CLASSES = 2
AUTH_CLASS_LABELS = ["fake", "real"]

# Metadata
CATEGORY = "bag-lv"
PART = "P07"

# Package name
PACKAGE_NAME = "lv_p07_v1_7"

# YOLO parameters
YOLO_CONF = 0.55
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (127, 127, 127)