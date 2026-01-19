"""
Configuration for Rolex P00 Models Version 1.3
"""
from raiki_sdk.utils import download_file_from_hf

# Model version
CLASSIFIER_MODEL_VERSION = "v1_3"
YOLO_MODEL_VERSION = "v1_3"

# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/watch_brand_detector_yolo",
    filename="watch_brand_detector_yolov11l_15122025.pt"
)

CLASSIFIER_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/om_p00_classify_v1-2",
    filename="om_p00_classify_v1.2_dinov3-arcface_20260112_20260112.pth"
)

EMBEDDING_STORAGE_PKL = download_file_from_hf(
    repo_id="rikai-ai/om_p00_classify_v1-2",
    filename="om_p00_classify_v1.2_dinov3-arcface_vector-storage_20260119_20260119_v2.pkl"
)

# Classifier model parameters
TARGET_SIZE = 256
CLASSIFIER_MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"
CLASSIFIER_NUM_CLASSES = 83
CLASSIFIER_EMBEDDING_SIZE = 512


# Metadata
CATEGORY = "om"
PART = "P00"

# Package name
PACKAGE_NAME = "om_p00_v1_2"

# YOLO parameters
YOLO_CONF = 0.55
YOLO_IMGSZ = 640
PADD_CROP = 0.05

# Resize parameters
COLOR_PADDING = (255, 255, 255)