"""
Configuration for Omega P03 Model Version 1.3
"""
from raiki_sdk.utils import download_file_from_hf
# Model version
AUTH_MODEL_VERSION = "v1.3" 
YOLO_MODEL_VERSION = "v1"


# convnext small 384
AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/omg_p03_auth_v1-3",
    filename="omg_p03_auth_v1-3_convnext_small_384_260115_260115.pth"
)

# YOLO weights
DETECTOR_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/omg_p03_detector_v1",
    filename="omg_p03_detector_v1_yolov11_251113_251115.pt"
)

# AUTH parameters
TARGET_SIZE = 384
AUTH_MODEL_NAME = "convnext_small.in12k_ft_in1k_384"
AUTH_NUM_CLASSES = 8
AUTH_CLASS_LABELS = ["black_fake", "black_real", "fake", "new_real", "old_real", "red_real", "white_fake", "white_real"]

# Metadata
CATEGORY = "omega"
PART = "P03"

# Package name
PACKAGE_NAME = "omg_p03_v1_3"

# YOLO parameters
YOLO_CONF = 0.5
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (255, 255, 255)