"""
Configuration for Louis Vuitton P00 Models Version 1.1
"""
from raiki_sdk.utils import download_file_from_hf

# Model version
CLASSIFY_MODEL_VERSION = "v1_4"
YOLO_MODEL_VERSION = "v1_1"

# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/brand_detector_yolo",
    filename="brand_detector_yolov11l_16122025.pt"
)

CLASSIFY_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/lv_p00_classify_v1-4",
    filename="lv_p00_classify_v1.4_dinov3-arcface_20250106_20250106.pth"
)

VECTOR_STORAGE_PKL = download_file_from_hf(
    repo_id="rikai-ai/lv_p00_classify_v1-4",
    filename="lv_p00_classify_v1.4_dinov3-arcface_vector-storage_20250106_20251206_v1.pkl"
)

# AUTH parameters
TARGET_SIZE = 256
CLASSIFY_MODEL_NAME = "DINOV3_ARCFACE_EBD"
NUM_CLASSES = 210
EMBEDDING_SIZE = 512

# CLASSIFY_CLASS_LABELS = ['fake','real']


# Metadata
CATEGORY = "bag-lv"
PART = "P00"

# Package name
PACKAGE_NAME = "lv_p00_v1_4"

# YOLO parameters
YOLO_CONF = 0.7
YOLO_IMGSZ = 640
PADD_CROP = 0.05
YOLO_INDEX = 3