"""
redframe_service.py
===================
RedFrameService — drop-in replacement for get_redframe_service().

Interface remains the same so the router does not need to change:
    redframe_service = RedFrameService(category, part, device, threshold)
    redframe_result  = redframe_service.forward(image, frame, position, crop_method)

    redframe_result.result        → bool
    redframe_result.error_type    → str | None
    redframe_result.iqa_result    → IqaResult | None
        .iqa_result.iqa_score
        .iqa_result.image_quality_validate
"""

from __future__ import annotations

import logging
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Callable, List

from PIL import Image
from raiki_sdk.base import IqaResult
from .iqa_service import LIQEServices

from .part_config import PART_CONFIGS, get_config, PartConfig
from .overlap_metrics import compute_overlap

log = logging.getLogger("RedFrameService")


# ══════════════════════════════════════════════════════════════
# Result dataclass  (keeps the same interface with the router)
# ══════════════════════════════════════════════════════════════

@dataclass
class RedFrameResult:
    result:     bool                  # True = Pass, False = Fail
    error_type: Optional[str] = None  # None when passed
    iqa_result: Optional[IqaResult] = None

    # Error type constants — aligned with the current router
    ERR_OUT_OF_FRAME = "out_of_frame_error"   # bbox is completely outside redframe
    ERR_WIDE         = "wide_error"            # bbox is too small relative to redframe
    ERR_ANGLE        = "angle_error"           # bbox angle error (reserved)
    ERR_CLOSE        = "close_error"           # bbox too close to edge (reserved)
    ERR_IQA          = "iqa_error"             # IQA failed
    ERR_NO_FRAME     = "no_frame_error"        # frame string is None or invalid


# ══════════════════════════════════════════════════════════════
# Frame string parser  ("x1,y1,x2,y2" normalized 0..1)
# ══════════════════════════════════════════════════════════════

def _parse_frame(frame_str: str, img_w: int, img_h: int) -> Optional[List[float]]:
    """
    Parse frame string "x1,y1,x2,y2" (normalized 0..1)
    into pixel coordinates [x1, y1, x2, y2].

    The router passes values like "0,0,1,1" or "0.1,0.2,0.8,0.9".
    Returns None if parsing fails.
    """
    try:
        parts = [float(v) for v in frame_str.strip().split(",")]
        if len(parts) != 4:
            return None
        x1n, y1n, x2n, y2n = parts
        return [x1n * img_w, y1n * img_h,
                x2n * img_w, y2n * img_h]
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════
# Core service
# ══════════════════════════════════════════════════════════════

class RedFrameService:
    """
    Validates the overlap between a YOLO bbox and the predefined redframe,
    then runs IQA on the cropped region.

    Parameters
    ----------
    category      : "rolex" | "omg" | "cartier"
    part          : "P01" | "P02" | ...
    device        : "cuda" | "cpu"
    threshold     : IQA score threshold
    advanced_mode : True = use custom LIQE model, False = use pyiqa standard
    """

    # Class-level IQA cache — prevents reloading on every instantiation
    _iqa_standard: Optional[LIQEServices] = None
    _iqa_advanced: Optional[LIQEServices] = None

    def __init__(self,
                 category: str,
                 part: str,
                 device: str = "cuda",
                 threshold: float = 0.1,
                 advanced_mode: bool = False):
        self.category      = category.lower()
        self.part          = part.upper()
        self.device        = device
        self.threshold     = threshold
        self.advanced_mode = advanced_mode

        # Resolve part_id key e.g. "rolex_p01", "omg_p02", "ctr_p01-1"
        self.part_id = self._resolve_part_id(category, part)

        # Lazy-load IQA singleton
        self._init_iqa()

    # ──────────────────────────────────────────────────────────
    # Setup
    # ──────────────────────────────────────────────────────────

    def _resolve_part_id(self, category: str, part: str) -> Optional[str]:
        """
        Map (category, part) to a part_id key in PART_CONFIGS.
        Examples:
            ("rolex",   "P01") → "rolex_p01"
            ("omg",     "P02") → "omg_p02"
            ("cartier", "P01") → tries "ctr_p01-1" then "ctr_p01-2"
        """
        cat_map = {
            "rolex":   "rolex",
            "omg":     "omg",
            "omega":   "omg",
            "cartier": "ctr",
            "ctr":     "ctr",
        }
        prefix = cat_map.get(category.lower())
        if prefix is None:
            log.warning(f"Category '{category}' not found in category mapping.")
            return None

        part_num  = part.lower().strip()          # "P01" → "p01"
        candidate = f"{prefix}_{part_num}"

        # Exact match
        if candidate in PART_CONFIGS:
            return candidate

        # Try numeric suffixes e.g. ctr_p01-1, ctr_p01-2
        for suffix in ["-1", "-2", "-3"]:
            if f"{candidate}{suffix}" in PART_CONFIGS:
                return f"{candidate}{suffix}"

        log.warning(f"No part_id found for ({category}, {part}). "
                    f"Candidate tried: '{candidate}'")
        return None

    def _init_iqa(self):
        """Initialize LIQEServices as a singleton per mode."""
        if self.advanced_mode:
            if RedFrameService._iqa_advanced is None:
                RedFrameService._iqa_advanced = LIQEServices(
                    device=self.device,
                    threshold=self.threshold,
                    use_advance_iqa=True,
                )
            self._iqa = RedFrameService._iqa_advanced
        else:
            if RedFrameService._iqa_standard is None:
                RedFrameService._iqa_standard = LIQEServices(
                    device=self.device,
                    threshold=self.threshold,
                    use_advance_iqa=False,
                )
            self._iqa = RedFrameService._iqa_standard

        # Update threshold — each part may have a different threshold
        self._iqa.threshold = self.threshold

    # ──────────────────────────────────────────────────────────
    # Crop helpers
    # ──────────────────────────────────────────────────────────

    def crop_method_default(self,
                             image: np.ndarray,
                             position: List[float]) -> np.ndarray:
        """Crop image by bbox [x1, y1, x2, y2]."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in position]
        return image[max(0, y1):min(h, y2),
                     max(0, x1):min(w, x2)].copy()

    def _crop_by_redframe(self, image: np.ndarray,
                           cfg: PartConfig) -> np.ndarray:
        """Crop image using the redframe coordinates (not the YOLO bbox)."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in cfg.redframe]
        return image[max(0, y1):min(h, y2),
                     max(0, x1):min(w, x2)].copy()

    # ──────────────────────────────────────────────────────────
    # Main entry point
    # ──────────────────────────────────────────────────────────

    def forward(self,
                image: np.ndarray,
                frame: Optional[str] = None,
                position: Optional[List[float]] = None,
                crop_method: Optional[Callable] = None) -> RedFrameResult:
        """
        Run overlap check then IQA on the cropped region.

        Args:
            image       : BGR numpy array from camera or file upload
            frame       : normalized frame string "x1,y1,x2,y2" from the router
            position    : YOLO bbox [x1,y1,x2,y2] in pixel coordinates
                          (from detector.validate().position after merge)
            crop_method : callable crop(image, position) → np.ndarray
                          None = use crop_method_default

        Returns:
            RedFrameResult
        """
        h, w = image.shape[:2]

        # ── Step 1: Resolve part config ───────────────────────
        if self.part_id is None:
            # No config found — skip overlap check and run IQA only
            return self._run_iqa_only(image, position, crop_method)

        try:
            cfg = get_config(self.part_id)
        except KeyError:
            log.warning(f"part_id '{self.part_id}' not found in PART_CONFIGS.")
            return self._run_iqa_only(image, position, crop_method)

        # ── Step 2: Validate frame string ─────────────────────
        if frame is not None:
            frame_bbox = _parse_frame(frame, w, h)
            if frame_bbox is None:
                log.warning(f"[{self.part_id}] Failed to parse frame string: '{frame}'")
                return RedFrameResult(
                    result=False,
                    error_type=RedFrameResult.ERR_NO_FRAME,
                )
        else:
            frame_bbox = None

        # ── Step 3: Validate position (YOLO bbox) ─────────────
        if position is None:
            log.warning(f"[{self.part_id}] Position is None.")
            return RedFrameResult(
                result=False,
                error_type=RedFrameResult.ERR_OUT_OF_FRAME,
            )

        yolo_bbox = position  # [x1, y1, x2, y2]

        # ── Step 4: Overlap check ─────────────────────────────
        overlap_result = self._check_overlap(yolo_bbox, cfg, w, h)
        if overlap_result is not None:
            # Return overlap error immediately without running IQA
            return overlap_result

        # ── Step 5: Crop + IQA ────────────────────────────────
        crop_fn = crop_method or self.crop_method_default
        crop    = crop_fn(image, yolo_bbox)

        if crop is None or crop.size == 0:
            log.warning(f"[{self.part_id}] Crop result is empty.")
            return RedFrameResult(
                result=False,
                error_type=RedFrameResult.ERR_OUT_OF_FRAME,
            )

        iqa_result = self._run_iqa(crop)

        if not iqa_result.image_quality_validate:
            log.info(f"[{self.part_id}] IQA FAIL  "
                     f"score={iqa_result.iqa_score:.4f}  "
                     f"threshold={iqa_result.iqa_threshold}")
            return RedFrameResult(
                result=False,
                error_type=RedFrameResult.ERR_IQA,
                iqa_result=iqa_result,
            )

        log.info(f"[{self.part_id}] PASS  score={iqa_result.iqa_score:.4f}")
        return RedFrameResult(result=True, iqa_result=iqa_result)

    # ──────────────────────────────────────────────────────────
    # Overlap logic
    # ──────────────────────────────────────────────────────────

    def _check_overlap(self,
                        yolo_bbox: List[float],
                        cfg: PartConfig,
                        img_w: int,
                        img_h: int) -> Optional[RedFrameResult]:
        """
        Check YOLO bbox against the redframe.
        Returns a RedFrameResult error if the check fails, or None if it passes.
        """
        metrics = compute_overlap(yolo_bbox, cfg)

        log.debug(f"[{self.part_id}] "
                  f"IoM={metrics['iom']:.3f}  "
                  f"deviation={metrics['deviation_pct']:.1f}%  "
                  f"tolerance={cfg.tolerance*100:.0f}%  "
                  f"passed={metrics['passed']}")

        if metrics["iom"] == 0.0:
            # No intersection — bbox is completely outside the redframe
            return RedFrameResult(
                result=False,
                error_type=RedFrameResult.ERR_OUT_OF_FRAME,
            )

        if not metrics["passed"]:
            # Partial intersection but below threshold — bbox too small or shifted
            return RedFrameResult(
                result=False,
                error_type=RedFrameResult.ERR_WIDE,
            )

        return None  # Overlap check passed

    # ──────────────────────────────────────────────────────────
    # IQA helpers
    # ──────────────────────────────────────────────────────────

    def _run_iqa(self, crop_bgr: np.ndarray) -> IqaResult:
        """Run LIQE IQA on a BGR crop."""
        pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        return self._iqa.iqa_detect(pil_img)

    def _run_iqa_only(self,
                       image: np.ndarray,
                       position: Optional[List[float]],
                       crop_method: Optional[Callable]) -> RedFrameResult:
        """
        Fallback when no part config exists — run IQA only without overlap check.
        Keeps backward compatibility for parts that have no redframe defined.
        """
        if position is None:
            return RedFrameResult(
                result=False,
                error_type=RedFrameResult.ERR_OUT_OF_FRAME,
            )
        crop_fn = crop_method or self.crop_method_default
        crop    = crop_fn(image, position)
        if crop is None or crop.size == 0:
            return RedFrameResult(
                result=False,
                error_type=RedFrameResult.ERR_OUT_OF_FRAME,
            )
        iqa_result = self._run_iqa(crop)
        return RedFrameResult(
            result=iqa_result.image_quality_validate,
            error_type=None if iqa_result.image_quality_validate else RedFrameResult.ERR_IQA,
            iqa_result=iqa_result,
        )