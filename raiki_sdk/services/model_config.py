"""
model_config.py
===============
Mapping from part_id → HuggingFace repository containing the corresponding .pt file.

Expected HuggingFace repository structure:
    username/ctr_p01-1
    └── best.pt   (or another filename specified via filename)

Usage:
    from model_config import HF_MODEL_CONFIGS, get_model_config
    cfg = get_model_config("rolex_p01")
    # cfg = {"repo_id": "username/rolex_p01", "filename": "best.pt", ...}
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class HFModelConfig:
    part_id: str
    repo_id: str                    # HuggingFace repository ID: "username/repo-name"
    filename: str                   # Model filename inside the repository
    revision: Optional[str] = None  # Branch/tag/commit hash, None = main


# ══════════════════════════════════════════════════════════════
# MODEL MAPPING — replace repo_id with your actual repositories
# ══════════════════════════════════════════════════════════════

HF_MODEL_CONFIGS: dict[str, HFModelConfig] = {

    # ── Cartier ───────────────────────────────────────────────
    "ctr_p01-1": HFModelConfig(
        part_id="ctr_p01-1",
        repo_id="rikai-ai/ctr_p01_detector_v1-1",
        filename="ctr_p01_detector_v1-1_20260206_20260206.pt",
    ),
    "ctr_p01-2": HFModelConfig(
        part_id="ctr_p01-2",
        repo_id="rikai-ai/ctr_p01_detector_v1-1",
        filename="ctr_p01_detector_v1-1_20260206_20260206.pt",
    ),
    "ctr_p02-1": HFModelConfig(
        part_id="ctr_p02-1",
        repo_id="rikai-ai/omg_p02_detector_v1",
        filename="omg_p02_detector_v1_rfdetr_20251125_20251125.pth",
    ),
    "ctr_p02-2": HFModelConfig(
        part_id="ctr_p02-2",
        repo_id="rikai-ai/omg_p02_detector_v1",
        filename="omg_p02_detector_v1_rfdetr_20251125_20251125.pth",
    ),
    "ctr_p03-1": HFModelConfig(
        part_id="ctr_p03-1",
        repo_id="rikai-ai/ctr_p03_detector_v1-1",
        filename="ctr_p03_detector_v1-1_20260212_20260212.pt",
    ),
    "ctr_p03-2": HFModelConfig(
        part_id="ctr_p03-2",
        repo_id="rikai-ai/ctr_p03_detector_v1-1",
        filename="ctr_p03_detector_v1-1_20260212_20260212.pt",
    ),
    "ctr_p04": HFModelConfig(
        part_id="ctr_p04",
        repo_id="rikai-ai/ctr_p04_detector_v1-1",
        filename="ctr_p04_detector_v1-1_20260206_20260206.pt"
    ),
    "ctr_p05": HFModelConfig(
        part_id="ctr_p05",
        repo_id="rikai-ai/ctr_p05_detector_v1-1",
        filename="ctr_p05_detector_v1-1_20260206_20260206.pt"
    ),
    "ctr_p06": HFModelConfig(
        part_id="ctr_p06",
        repo_id="rikai-ai/ctr_p06_detector_v1-1",
        filename="ctr_p06_detector_v1-1_20260206_20260206.pt"
    ),

    # ── Omega ─────────────────────────────────────────────────
    "omg_p01": HFModelConfig(
        part_id="omg_p01",
        repo_id="rikai-ai/omg_p01_detector_v1",
        filename="omg_p01_detector_v1_yolo11_20251114_20251117.pt"
    ),
    "omg_p02": HFModelConfig(
        part_id="omg_p02",
        repo_id="rikai-ai/omg_p02_detector_v1",
        filename="omg_p02_detector_v1_rfdetr_20251125_20251125.pth"
    ),
    "omg_p03": HFModelConfig(
        part_id="omg_p03",
        repo_id="rikai-ai/omg_p03_detector_v1",
        filename="omg_p03_detector_v1_yolov11_251113_251115.pt"
    ),
    "omg_p05": HFModelConfig(
        part_id="omg_p05",
        repo_id="rikai-ai/omg_p05_detector_v1-4",
        filename="omg_p05_detector_v1-406032026-06032026.pt"
    ),
    "omg_p06": HFModelConfig(
        part_id="omg_p06",
        repo_id="rikai-ai/omg_p06_detector_v1-2",
        filename="omg_p06_detector_v1-2_yolo11s_260603_260603_v2.pt"
    ),

    # ── Rolex ─────────────────────────────────────────────────
    "rolex_p01": HFModelConfig(
        part_id="rolex_p01",
        repo_id="rikai-ai/rolex_p01_detector_v1-3",
        filename="rolex_p01_detector_v1-3_yolov11m_241226_241226.pt"
    ),
    "rolex_p02": HFModelConfig(
        part_id="rolex_p02",
        repo_id="rikai-ai/rolex_p02_detector_v1-4",
        filename="rolex_p02_detector_v1-4_yolov11_260415_260415.pt"
    ),
    "rolex_p03": HFModelConfig(
        part_id="rolex_p03",
        repo_id="rikai-ai/rolex_p03_detector_v1-3",
        filename="rolex_p03_detector_v1-3_yolov11m_241129_251129.pt"
    ),
    "rolex_p04": HFModelConfig(
        part_id="rolex_p04",
        repo_id="rikai-ai/rolex_p04_detector_v1_3",
        filename="rolex_p04_detector_v1_3_yolov11m_241203_241204.pt"
    ),
    "rolex_p05": HFModelConfig(
        part_id="rolex_p05",
        repo_id="rikai-ai/rolex_p05_detector_v1-3",
        filename="rolex_p05_detector_v1-3_yolo_241202_241203.pt"
    ),
    "rolex_p08": HFModelConfig(
        part_id="rolex_p08",
        repo_id="rikai-ai/rolex_p08_detector_v1-4",
        filename="rolex_p08_detector_v1-4_yolov11_260421_260421.pt"
    ),
    "rolex_p10": HFModelConfig(
        part_id="rolex_p10",
        repo_id="rikai-ai/rolex_p10_detector_v1",
        filename="rolex_p10_detector_v1_yolo_251005_251006.pt"
    ),
}


def get_model_config(part_id: str) -> HFModelConfig:
    if part_id not in HF_MODEL_CONFIGS:
        raise KeyError(
            f"Part '{part_id}' does not exist in HF_MODEL_CONFIGS. "
            f"Please verify model_config.py."
        )
    return HF_MODEL_CONFIGS[part_id]