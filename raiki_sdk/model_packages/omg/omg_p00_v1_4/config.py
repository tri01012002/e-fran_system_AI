"""
Configuration for Omega P00 Models Version 1.4
"""
from raiki_sdk.utils import download_file_from_hf

# Model version
CLASSIFIER_MODEL_VERSION = "v1_4"
YOLO_MODEL_VERSION = "v1_3"

# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/watch_brand_detector_yolo",
    filename="watch_brand_detector_yolov11l_15122025.pt"
)

CLASSIFIER_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/om_p00_classify_v1-4",
    filename="om_p00_classify_v1.4_dinov3-arcface_121classes_20260409_20260409.pth"
)

EMBEDDING_STORAGE_PKL = download_file_from_hf(
    repo_id="rikai-ai/om_p00_classify_v1-4",
    filename="om_p00_classify_v1.4_dinov3-arcface_vector-storage_20260409_20260409.pkl"
)

# Classifier model parameters
TARGET_SIZE = 256
CLASSIFIER_MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"
CLASSIFIER_NUM_CLASSES = 121
CLASSIFIER_EMBEDDING_SIZE = 512


# Metadata
CATEGORY = "om"
PART = "P00"

# Package name
PACKAGE_NAME = "om_p00_v1_4"

# YOLO parameters
YOLO_CONF = 0.55
YOLO_IMGSZ = 640
PADD_CROP = 0.05

# Resize parameters
COLOR_PADDING = (255, 255, 255)