from raiki_sdk.utils.hf_lib import download_file_from_hf

MODEL_VERSION = "v1.0"
CLASSIFY_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/brand_classify_v1",
    filename="brand_classify_v1_convnext_small_384_20251217_20251217.pth",
)

TARGET_SIZE = 384
MODEL_NAME = "convnext_small.in12k_ft_in1k_384"
NUM_CLASSES = 2
CLASS_LABELS = ["omg", "rolex"]

CONFIDENCE_THRESHOLD = 0.35