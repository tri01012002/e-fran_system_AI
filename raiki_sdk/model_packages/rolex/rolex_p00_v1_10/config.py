"""
Configuration for Rolex P00 Models Version 1.10
"""
from raiki_sdk.utils import download_file_from_hf

# Model version
CLASSIFIER_MODEL_VERSION = "v1_10"
YOLO_MODEL_VERSION = "v1_3"

# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex_p00_detector_v1-3",
    filename="rolex_p00_detector_v1-3_yolov11_20250529_20250530.pt"
)

CLASSIFIER_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex_p00_classify_aug26_350",
    filename="classification_rolex_p00_aug_350.pth"
)

EMBEDDING_STORAGE_PKL = download_file_from_hf(
    repo_id="rikai-ai/rolex_p00_classify_aug26_350",
    filename="rolex_p00_vit-arcface_vector-storage_v5.pkl"
)

# Classifier model parameters
TARGET_SIZE = 256
CLASSIFIER_MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"
CLASSIFIER_NUM_CLASSES = 350
CLASSIFIER_EMBEDDING_SIZE = 512


# Metadata
CATEGORY = "rolex"
PART = "P00"

# Package name
PACKAGE_NAME = "rolex_p00_v1_10"

# YOLO parameters
YOLO_CONF = 0.55
YOLO_IMGSZ = 640
PADD_CROP = 0.05

# Resize parameters
COLOR_PADDING = (255, 255, 255)