"""
Configuration for Louis Vuitton P00 Models Version 1.1
"""
from raiki_sdk.utils import download_file_from_hf

# Model version
CLASSIFY_MODEL_VERSION = "v1_5"
YOLO_MODEL_VERSION = "v1_1"

# Model weights paths (relative to package root or absolute)
# YOLO_WEIGHTS = download_file_from_hf(
#     repo_id="rikai-ai/brand_detector_yolo",
#     filename="brand_detector_yolov11l_16122025.pt"
# )

CLASSIFY_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/lv_p00_classify_v1-5",
    filename="lv_p00_classify_v1.5_220classes_dinov3-arcface_20250209_20250209.pth"
)

VECTOR_STORAGE_PKL = download_file_from_hf(
    repo_id="rikai-ai/lv_p00_classify_v1-5",
    filename="lv_p00_classify_v1.5_dinov3-arcface_vector-storage_20260209_2020209_v1.pkl"
)

# AUTH parameters
TARGET_SIZE = 256
COLOR_PADDING = (255, 255, 255)
CLASSIFY_MODEL_NAME = "DINOV3_ARCFACE_EBD"
NUM_CLASSES = 220
EMBEDDING_SIZE = 512

# CLASSIFY_CLASS_LABELS = ['fake','real']


# Metadata
CATEGORY = "bag-lv"
PART = "P00"

# Package name
PACKAGE_NAME = "lv_p00_v1_5"

# YOLO parameters
YOLO_CONF = 0.7
YOLO_IMGSZ = 640
PADD_CROP = 0.05
YOLO_INDEX = 3