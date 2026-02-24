from raiki_sdk.utils.hf_lib import download_file_from_hf

MODEL_VERSION = "v1.1"
CLASSIFY_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/brand_classify_v1-1",
    filename="brand_classify_rolex_omg_ctr_convnext_small_v1.1.pth",
)

TARGET_SIZE = 384
MODEL_NAME = "convnext_small.in12k_ft_in1k_384"
NUM_CLASSES = 3
CLASS_LABELS = sorted(["omg", "rolex", "ctr"])

CONFIDENCE_THRESHOLD = 0.65