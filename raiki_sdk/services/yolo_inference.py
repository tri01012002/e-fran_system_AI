"""
yolo_inference.py
=================
Load YOLO models from HuggingFace Hub and run inference.

Flow:
    hf_hub_download(repo_id, filename)
        → cached to ~/.cache/huggingface/hub/...
        → YOLO(local_path).to(device)

First run: download and cache.
Subsequent runs: load from cache without downloading again.

Usage:
    from yolo_inference import YOLOModelManager

    manager = YOLOModelManager(device="cuda")
    manager.load_all()                        # download + load 28 models

    detections = manager.detect_pending(frame, pending_parts)
    # → [{"part_id": str, "bbox": [x1,y1,x2,y2], "conf": float}, ...]
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np

from model_config import HF_MODEL_CONFIGS, get_model_config, HFModelConfig

log = logging.getLogger("YOLOInference")


class YOLOModelManager:
    """
    Manage YOLO models loaded from HuggingFace Hub.

    Parameters
    ----------
    device         : "cuda" | "cuda:0" | "cpu"
    conf_threshold : minimum confidence threshold to accept detections
    hf_token       : HuggingFace token (for private repositories)
    cache_dir      : local cache directory, None = use HF default cache
    """

    def __init__(self,
                 device: str = "cuda",
                 conf_threshold: float = 0.45,
                 hf_token: Optional[str] = None,
                 cache_dir: Optional[str] = None):
        self.device         = device
        self.conf_threshold = conf_threshold
        self.hf_token       = hf_token
        self.cache_dir      = cache_dir

        self._models: Dict[str, object] = {}
        self._failed: List[str] = []

    # ──────────────────────────────────────────────────────────
    # Download + Load
    # ──────────────────────────────────────────────────────────

    def _download_model(self, cfg: HFModelConfig) -> str:
        """
        Download a .pt file from HuggingFace Hub into the local cache.
        Returns the local path to the downloaded .pt file.
        """
        from huggingface_hub import hf_hub_download

        log.debug(f"  Downloading {cfg.repo_id}/{cfg.filename}...")
        local_path = hf_hub_download(
            repo_id=cfg.repo_id,
            filename=cfg.filename,
            revision=cfg.revision,
            token=self.hf_token,
            cache_dir=self.cache_dir,
        )
        return local_path

    def _load_model(self, part_id: str) -> bool:
        """
        Download (if necessary) and load a single model.
        Returns True if successful.
        """
        from ultralytics import YOLO

        try:
            cfg = get_model_config(part_id)
            local_path = self._download_model(cfg)
            model = YOLO(local_path)
            model.to(self.device)
            self._models[part_id] = model
            log.debug(f"  ✓ {part_id} ({cfg.repo_id})")
            return True
        except Exception as e:
            self._failed.append(part_id)
            log.error(f"  ✗ {part_id}: {e}")
            return False

    def load_all(self, part_ids: Optional[List[str]] = None):
        """
        Load all models defined in HF_MODEL_CONFIGS.

        Parameters
        ----------
        part_ids : list[str] | None
            Only load models for the specified parts.
            If None, load all available models.
        """
        targets = part_ids or list(HF_MODEL_CONFIGS.keys())
        log.info(
            f"Loading {len(targets)} YOLO models from HuggingFace "
            f"on {self.device}..."
        )

        for part_id in targets:
            self._load_model(part_id)

        log.info(
            f"Loaded: {len(self._models)} OK  |  "
            f"Failed: {len(self._failed)}"
        )

        if self._failed:
            log.warning(f"Failed models: {self._failed}")

    def load_one(self, part_id: str):
        """Lazily load a single model by part_id."""
        if part_id not in self._models:
            self._load_model(part_id)

    # ──────────────────────────────────────────────────────────
    # Detection
    # ──────────────────────────────────────────────────────────

    def detect_one(self,
                   part_id: str,
                   frame: np.ndarray,
                   verbose: bool = False) -> Optional[dict]:
        """
        Run inference using a single model on the given frame.

        Returns:
            {"part_id": str, "bbox": [x1,y1,x2,y2], "conf": float}
            or None if no object is detected.
        """
        model = self._models.get(part_id)

        if model is None:
            log.warning(f"[{part_id}] Model has not been loaded.")
            return None

        results = model.predict(
            source=frame,
            device=self.device,
            conf=self.conf_threshold,
            verbose=verbose,
            stream=False,
        )

        best = _get_best_detection(results)

        if best is None:
            log.debug(f"[{part_id}] No detection found.")
            return None

        return {
            "part_id": part_id,
            "bbox": best["bbox"],
            "conf": best["conf"],
        }

    def detect_all(self,
                   frame: np.ndarray,
                   part_ids: Optional[List[str]] = None) -> List[dict]:
        """
        Run all (or selected) models on the same frame.
        """
        targets = part_ids if part_ids else list(self._models.keys())

        return [
            det
            for pid in targets
            if (det := self.detect_one(pid, frame)) is not None
        ]

    def detect_pending(self,
                       frame: np.ndarray,
                       pending_parts: set) -> List[dict]:
        """
        Run detection only for parts that have not yet been captured.

        Improves performance by skipping completed parts.
        """
        return self.detect_all(frame, part_ids=list(pending_parts))

    # ──────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────

    @property
    def loaded_parts(self) -> List[str]:
        return list(self._models.keys())

    def __len__(self):
        return len(self._models)

    def __repr__(self):
        return (
            f"YOLOModelManager("
            f"loaded={len(self._models)}, "
            f"failed={len(self._failed)}, "
            f"device={self.device})"
        )


# ══════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════

def _get_best_detection(results) -> Optional[dict]:
    """
    Extract the highest-confidence bounding box
    from Ultralytics prediction results.
    """
    if not results or len(results) == 0:
        return None

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return None

    confs = result.boxes.conf.cpu().numpy()
    best_idx = int(confs.argmax())
    bbox = result.boxes.xyxy[best_idx].cpu().numpy()

    return {
        "bbox": [
            float(bbox[0]),
            float(bbox[1]),
            float(bbox[2]),
            float(bbox[3]),
        ],
        "conf": round(float(confs[best_idx]), 4),
    }