"""
Configuration for LIQE Custom Services and Red Frame Services
"""
from raiki_sdk.config import FOLDER_CACHE_MODEL_HF
from raiki_sdk.utils.hf_lib import download_file_from_hf

# Model weights paths (relative to package root or absolute)
LIQE_CUSTOM_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/liqe_iqa_model", 
    filename="LIQE.pt"
)

# Xgboost Rolex model weights path
ROLEX_P01_XGB = download_file_from_hf(
    repo_id="rikai-ai/rolex_p01_redframe_v1", 
    filename="rolex_p01_redframe_v1_xgb_250915_251023.pkl"
)

ROLEX_P02_XGB = download_file_from_hf(
    repo_id="rikai-ai/rolex_p02_redframe_v1", 
    filename="rolex_p02_redframe_v1_xgb_250917_251023.pkl"
)

ROLEX_P03_XGB = download_file_from_hf(
    repo_id="rikai-ai/rolex_p03_redframe_v1", 
    filename="rolex_p03_redframe_v1_xgb_250918_251023.pkl"
)

# Helper function to get model path based on category and part
def get_redframe_model_path(category: str, part: str) -> str:
    """
    Get model path based on category and part
    Args:
        category: Category of the model (e.g. "rolex")
        part: Part of the model (e.g. "p01")
    Returns:
        str: Model path
    """
    MODEL_REGISTRY = {
        ("rolex", "p01"): ROLEX_P01_XGB,
        ("rolex", "p02"): ROLEX_P02_XGB,
        ("rolex", "p03"): ROLEX_P03_XGB,
    }
    key = (category.lower(), part.lower())
    model_path = MODEL_REGISTRY[key]
    if model_path is None:
        print(f"[INFO] Model không tồn tại cho {category} - {part}, skipping")
    return MODEL_REGISTRY[key]



# ============ BRAND DETECTION CONFIG ============
WATCH_DETECTOR_WEIGHTS = download_file_from_hf(
    repo_id="rikai-ai/brand_detector_v1.5",
    filename="brand_detector_rolex_omg_ctr_lv_yolo26_sizeL_20260224_v1.5.pt"  
)

WATCH_DETECTOR_IMGSZ = 640
WATCH_DETECTOR_CONFIDENCE = 0.8
WATCH_CLASS_MAP = {
    0: "watch",
    1: "omega_logo",
    2: "rolex_logo",
    3: "lv",
}

WATCH_BRAND_RULES = {
    "omega_logo": "omg",
    "rolex_logo": "rolex",
    "lv": "bag-lv",
}

LV_DEFAULT_MODEL_VERSION = "v1_1"