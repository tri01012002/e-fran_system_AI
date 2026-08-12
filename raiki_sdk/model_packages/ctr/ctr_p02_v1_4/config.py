"""
Configuration for CTR P02 Models Version 1_4
"""

from raiki_sdk.utils import download_file_from_hf
import cv2

# Model version
AUTH_MODEL_VERSION = "v1_4"
YOLO_MODEL_VERSION = "v1"

DETECTOR_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/ctr_P02_classify",
    filename="model_yolo_ctr_P02_070826.pt"
)

AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/ctr_P02_classify",
    filename="model_classify_ctr_P02_070826.pth"
)

# AUTH parameters
TARGET_SIZE = 512
AUTH_MODEL_NAME = "maxvit_small_tf_512.in1k"
AUTH_NUM_CLASSES = 2
AUTH_CLASS_LABELS = ['fake', 'real']

# Metadata
CATEGORY = "ctr"
PART = "P02"

# Package name
PACKAGE_NAME = "ctr_p02_v1_4"

# YOLO parameters
YOLO_CONF = 0.65
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (255, 255, 255)
