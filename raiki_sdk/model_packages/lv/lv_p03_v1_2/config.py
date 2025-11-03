"""
Configuration for Louis Vuitton P03 Models Version 1.2
"""
from raiki_sdk.utils import download_file_from_hf

# Model version
AUTH_MODEL_VERSION = "v1_2"
YOLO_MODEL_VERSION_1 = "v1_1"
YOLO_MODEL_VERSION_2 = "v1_2"

# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS_1 = download_file_from_hf(
    repo_id="rikai-ai/lv_p03_detector_v1-1",
    filename="lv_p03_detector_v1-1_yolov11_250424_250425.pt"
)

YOLO_WEIGHTS_2 = download_file_from_hf(
    repo_id="rikai-ai/lv_p03_detector_v1-2",
    filename="lv_p03_detector_v1-2_yolov11_250805_250806.pt"
)

AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/lv_p03_auth_v1-2",
    filename="lv_p03_v1-2_auth_cmal_251016_251017.pth"
)

# AUTH parameters
TARGET_SIZE = 421
AUTH_MODEL_NAME = "CMAL"
AUTH_NUM_CLASSES = 8
AUTH_CLASS_LABELS = ['fake_case1n3n5-8', 'fake_case2', 'fake_case4', 'fake_hook',
                  'real_case1n3n5-8', 'real_case2', 'real_case4', 'real_hook']


# Metadata
CATEGORY = "bag-lv"
PART = "P03"

# Package name
PACKAGE_NAME = "lv_p03_v1_2"

# YOLO parameters
YOLO_CONF = 0.50
YOLO_IMGSZ = 640

# Crop Parameters
PAD = 50