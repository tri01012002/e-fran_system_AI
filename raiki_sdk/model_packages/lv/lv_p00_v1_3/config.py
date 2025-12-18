"""
Configuration for Louis Vuitton P00 Models Version 1.1
"""
from raiki_sdk.utils import download_file_from_hf

# Model version
CLASSIFY_MODEL_VERSION = "v1_3"
YOLO_MODEL_VERSION = "v1_1"

# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/lv_p00_detector_v1-1",
    filename="lv_p00_detector_v1-1_yolov11_250523_250523.pt"
)

CLASSIFY_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/lv_p00_classify_v1-3",
    filename="lv_p00_classify_v1.3_dinov3-arcface_20251212_20251212.pth"
)

VECTOR_STORAGE_PKL = download_file_from_hf(
    repo_id="rikai-ai/lv_p00_classify_v1-3",
    filename="lv_p00_classify_v1.3_dinov3-arcface_vector-storage_20251212_20251212.pkl"
)

# AUTH parameters
TARGET_SIZE = 256
CLASSIFY_MODEL_NAME = "DINOV3_ARCFACE_EBD"
NUM_CLASSES = 190
EMBEDDING_SIZE = 512

# CLASSIFY_CLASS_LABELS = ['fake','real']


# Metadata
CATEGORY = "bag-lv"
PART = "P00"

# Package name
PACKAGE_NAME = "lv_p00_v1_3"

# YOLO parameters
YOLO_CONF = 0.7
YOLO_IMGSZ = 640
PADD_CROP = 0.05