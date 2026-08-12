
"""
Configuration for Rolex P03 Model
"""
from raiki_sdk.utils import download_file_from_hf
# Model version
AUTH_MODEL_VERSION = "v1_6"
YOLO_MODEL_VERSION = "v1_3"


# Model weights paths (relative to package root or absolute)
YOLO_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex-p03-fake-real-resnet50",
    filename="rolex_p03_yolo_best.pt"
)

AUTH_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/rolex-p03-fake-real-resnet50",
    filename="rolex_p03_classify_best_model.pth"
)

# AUTH parameters
TARGET_SIZE = 224
AUTH_MODEL_NAME = "ResNet50"
AUTH_NUM_CLASSES = 2
AUTH_CLASS_LABELS = [
    'real',
    'fake',
]

# Metadata
CATEGORY = "rolex"
PART = "P03"

# Package name
PACKAGE_NAME = "rolex_p03_v1_6"

# YOLO parameters
YOLO_CONF = 0.7
YOLO_IMGSZ = 640

# Resize parameters
COLOR_PADDING = (85, 85, 85)
RESIZE_OUTPUT_SIZE = 256