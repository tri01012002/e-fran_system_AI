"""
Configuration for Louis Vuitton P00 Models Version 1.1
"""
from raiki_sdk.utils import download_file_from_hf

# Model version
AUTH_MODEL_VERSION = "v1_1"
YOLO_MODEL_VERSION = "v1_1"

# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/lv_p00_detector_v1-1",
    filename="lv_p00_detector_v1-1_yolov11_250523_250523.pt"
)

AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/lv_p00_classify_v1-1",
    filename="lv_p00_classify_v1-1_maxvit-arcface-ebd_251010_251010.pth"
)

VECTOR_STORAGE_PKL = download_file_from_hf(
    repo_id="rikai-ai/lv_p00_classify_v1-1",
    filename="lv_p00_classify_v1-1_vector-storage_251010_251010.pkl"
)

# AUTH parameters
TARGET_SIZE = 384
AUTH_MODEL_NAME = "MAXVIT_ARCFACE_EBD"
NUM_CLASSES = 150
EMBEDDING_SIZE = 512

# AUTH_CLASS_LABELS = ['fake','real']


# Metadata
CATEGORY = "bag-lv"
PART = "P00"

# Package name
PACKAGE_NAME = "lv_p00_v1_1"

# YOLO parameters
YOLO_CONF = 0.55
YOLO_IMGSZ = 640
PADD_CROP = 0.05