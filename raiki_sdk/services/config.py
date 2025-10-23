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
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Model không tồn tại cho {category} - {part}")
    return MODEL_REGISTRY[key]