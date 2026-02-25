"""
Configuration for Rolex P00 Models Version 1.3
"""
from raiki_sdk.utils import download_file_from_hf

# Model version
CLASSIFIER_MODEL_VERSION = "v1_6"
YOLO_MODEL_VERSION = "v1_3"

# Model weights paths (relative to package root or absolute)
# YOLO_WEIGHTS = download_file_from_hf(
#     repo_id="rikai-ai/rolex_p00_detector_v1-3",
#     filename="rolex_p00_detector_v1-3_yolov11_20250529_20250530.pt"
# )

CLASSIFIER_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex_p00_classify_v1-6",
    filename="rolex_p00_classify_v1.6_272classes_dinov3-arceface-vector-50gray_20260225_20260225.pth"
)

EMBEDDING_STORAGE_PKL = download_file_from_hf(
    repo_id="rikai-ai/rolex_p00_classify_v1-6",
    filename="rolex_p00_classify_1.6_272classes_dinov3-arceface-vector-withgray-storage_20250225_20250225.pkl"
)

# Classifier model parameters
TARGET_SIZE = 256
CLASSIFIER_MODEL_NAME = "vit_base_patch16_dinov3.lvd1689m"
CLASSIFIER_NUM_CLASSES = 272
CLASSIFIER_EMBEDDING_SIZE = 512


# Metadata
CATEGORY = "rolex"
PART = "P00"

# Package name
PACKAGE_NAME = "rolex_p00_v1_6"

# YOLO parameters
YOLO_CONF = 0.55
YOLO_IMGSZ = 640
PADD_CROP = 0.05

# Resize parameters
COLOR_PADDING = (255, 255, 255)