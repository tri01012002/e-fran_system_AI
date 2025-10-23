from huggingface_hub import hf_hub_download
from raiki_sdk.config import FOLDER_CACHE_MODEL_HF

def download_file_from_hf(repo_id: str, filename: str) -> str:

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=FOLDER_CACHE_MODEL_HF
    )
