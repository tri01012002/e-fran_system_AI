"""
Configuration for LIQE Custom Services and Red Frame Services
"""
from raiki_sdk.config import MODEL_DIR
from huggingface_hub import hf_hub_download

# Model weights paths (relative to package root or absolute)
LIQE_CUSTOM_WEIGHTS = hf_hub_download(
    repo_id="rikai-ai/liqe_iqa_model", 
    filename="LIQE.pt", 
    cache_dir=MODEL_DIR)

# Xgboost Rolex model weights path
ROLEX_P01_XGB = hf_hub_download(
    repo_id="rikai-ai/rolex_p01_redframe_v1", 
    filename="rolex_p01_redframe_v1_xgb_250915.pkl", 
    cache_dir=MODEL_DIR)

ROLEX_P02_XGB = hf_hub_download(
    repo_id="rikai-ai/rolex_p02_redframe_v1", 
    filename="rolex_p02_redframe_v1_xgb_250917.pkl", 
    cache_dir=MODEL_DIR)

ROLEX_P03_XGB = hf_hub_download(
    repo_id="rikai-ai/rolex_p03_redframe_v1", 
    filename="rolex_p03_redframe_v1_xgb_250918.pkl", 
    cache_dir=MODEL_DIR)

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