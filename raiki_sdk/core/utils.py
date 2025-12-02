import yaml
import os
import shutil
import logging
from pathlib import Path

_logger = logging.getLogger(__name__)

def _normalize_version_str(version: str) -> str:
    """Ensure version normalization
    # e.g. - "v1.3" => "v1_3"
    """
    try:
        if "." in version and "v" in version.lower():
            return version.replace(".", "_")
        elif "v" in version.lower() and "_" in version:
            return version  # Already normalized
        elif "v" in version.lower():
            return version  
        else:
            raise ValueError(f"Error normalizing version '{version}'")
    except Exception as e:
        raise ValueError(f"Error normalizing version '{version}': {e}") from e

def _normalize_part_str(part: str) -> str:
    """Ensure part normalization
    # e.g. - "LV_P01" => "p01", "ROLEX_P01" => "p01"
    """
    try:
        if "_" in part:
            return part.split("_")[1].lower()
        else:
            return part
    except Exception as e:
        raise ValueError(f"Error normalizing part '{part}': {e}")

def _normalize_category_str(category: str) -> str:
    """Ensure category normalization
    "bag-lv" => "lv"
    "rolex" => "rolex"
    """
    try:
        if "bag-" in category.lower():
            return category.replace("bag-", "")
        elif "rolex" in category.lower():
            return "rolex"
        elif "omg" in category.lower():
            return "omg"
        else:
            raise ValueError(f"Error normalizing category '{category}'")
    except Exception as e:
        raise ValueError(f"Error normalizing category '{category}': {e}")