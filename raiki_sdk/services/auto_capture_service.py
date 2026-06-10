"""
auto_capture_service.py
=======================
AutoCaptureService — runs in the background, continuously reads frames
from the camera, and automatically triggers capture when:
    1. YOLO detects the part in the frame
    2. The detected bbox overlaps the redframe by >= (1 - tolerance)
    3. IQA score passes the threshold

This plugs directly into the existing stack:
    - Uses the same RedFrameService.forward() already wired in the router
    - Uses the same YOLOModelManager already loading models from HuggingFace
    - Calls on_capture callback (or pushes via WebSocket) when a capture succeeds

Usage (from your FastAPI startup):
    from auto_capture_service import AutoCaptureService

    service = AutoCaptureService(
        category="rolex",
        parts=["P01", "P02", "P03"],
        camera_source=0,
        device="cuda",
        on_capture=your_callback,   # called with CaptureResult on each success
    )
    await service.start()   # non-blocking, runs in background thread
    ...
    await service.stop()
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import cv2
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, List, Dict

log = logging.getLogger("AutoCaptureService")


# ══════════════════════════════════════════════════════════════
# Result dataclass
# ══════════════════════════════════════════════════════════════

@dataclass
class CaptureResult:
    part_id:    str
    category:   str
    frame_id:   int
    iqa_score:  float
    iqa_threshold: float
    bbox:       List[float]
    saved_path: Optional[str] = None
    timestamp:  float = field(default_factory=time.time)

    def __str__(self):
        return (f"[{self.category}/{self.part_id}] "
                f"frame={self.frame_id} "
                f"IoA score={self.iqa_score:.4f} "
                f"→ {Path(self.saved_path).name if self.saved_path else 'not saved'}")


# ══════════════════════════════════════════════════════════════
# State tracker per part
# ══════════════════════════════════════════════════════════════

@dataclass
class PartState:
    part_id:     str
    captured:    bool = False
    retry_count: int  = 0
    last_iom:    float = 0.0
    last_iqa:    float = 0.0


# ══════════════════════════════════════════════════════════════
# AutoCaptureService
# ══════════════════════════════════════════════════════════════

class AutoCaptureService:
    """
    Background service that continuously reads from a camera and
    automatically captures each part when overlap + IQA both pass.

    Parameters
    ----------
    category       : "rolex" | "omg" | "cartier"
    parts          : list of part codes to capture, e.g. ["P01","P02","P08"]
    camera_source  : int (device index) or str (RTSP URL)
    device         : "cuda" | "cpu"
    max_retries    : max IQA retry attempts per part before giving up
    save_dir       : directory to save captured images (None = do not save)
    on_capture     : callback(CaptureResult) called on each successful capture
    frame_skip     : process every N-th frame (1 = every frame)
    """

    def __init__(self,
                 category: str,
                 parts: List[str],
                 camera_source=0,
                 device: str = "cuda",
                 max_retries: int = 3,
                 save_dir: Optional[str] = "captures/",
                 on_capture: Optional[Callable[[CaptureResult], None]] = None,
                 frame_skip: int = 1):

        self.category      = category
        self.parts         = [p.upper() for p in parts]
        self.camera_source = camera_source
        self.device        = device
        self.max_retries   = max_retries
        self.save_dir      = Path(save_dir) if save_dir else None
        self.on_capture    = on_capture
        self.frame_skip    = frame_skip

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # State
        self._states: Dict[str, PartState] = {
            p: PartState(part_id=p) for p in self.parts
        }
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame_id = 0

        # Lazy-loaded services (initialized on first start)
        self._yolo_manager  = None
        self._redframe_svcs: Dict[str, object] = {}

    # ──────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────

    async def start(self):
        """Start the capture loop in a background thread (non-blocking)."""
        log.info(f"Starting AutoCaptureService: "
                 f"category={self.category} parts={self.parts}")
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="AutoCaptureLoop",
            daemon=True,
        )
        self._thread.start()

    async def stop(self):
        """Signal the loop to stop and wait for the thread to exit."""
        log.info("Stopping AutoCaptureService...")
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        log.info("AutoCaptureService stopped.")

    def reset(self):
        """Reset all part states (call when starting a new product)."""
        for state in self._states.values():
            state.captured    = False
            state.retry_count = 0
            state.last_iom    = 0.0
            state.last_iqa    = 0.0
        log.info(f"Reset {len(self._states)} part states.")

    # ──────────────────────────────────────────────────────────
    # Lazy init
    # ──────────────────────────────────────────────────────────

    def _init_yolo(self):
        """Load YOLO models once on first use."""
        from .yolo_inference import YOLOModelManager
        from .part_config import PART_CONFIGS

        # Map part codes (P01) to part_ids (rolex_p01) to know which models to load
        part_ids = []
        for part in self.parts:
            pid = self._resolve_part_id(part)
            if pid:
                part_ids.append(pid)

        self._yolo_manager = YOLOModelManager(device=self.device)
        self._yolo_manager.load_all(part_ids=part_ids)

    def _get_redframe_service(self, part: str):
        """Get or create a RedFrameService for a part (cached)."""
        if part not in self._redframe_svcs:
            from .redframe_service_fix import RedFrameService
            from config import settings
            from routers.utils import get_model_setting, get_part_name

            part_name = get_part_name(part, self.category)
            threshold = get_model_setting(self.category, part_name, "threshold")

            advanced_parts = getattr(settings, "advanced_redframe_parts", {})
            advanced_mode  = (
                getattr(settings, "use_advanced_redframe", False)
                and part in advanced_parts.get(self.category, [])
            )

            self._redframe_svcs[part] = RedFrameService(
                category=self.category,
                part=part,
                device=self.device,
                threshold=threshold,
                advanced_mode=advanced_mode,
            )
        return self._redframe_svcs[part]

    def _resolve_part_id(self, part: str) -> Optional[str]:
        """Map part code (P01) to part_id (rolex_p01) used in PART_CONFIGS."""
        from .part_config import PART_CONFIGS
        cat_prefix = {
            "rolex": "rolex", "omg": "omg", "omega": "omg",
            "cartier": "ctr", "ctr": "ctr",
        }.get(self.category.lower())
        if not cat_prefix:
            return None
        candidate = f"{cat_prefix}_{part.lower()}"
        if candidate in PART_CONFIGS:
            return candidate
        for suffix in ["-1", "-2", "-3"]:
            if f"{candidate}{suffix}" in PART_CONFIGS:
                return f"{candidate}{suffix}"
        return None

    # ──────────────────────────────────────────────────────────
    # Main loop
    # ──────────────────────────────────────────────────────────

    def _run_loop(self):
        """Main camera loop — runs in background thread."""
        # Initialize YOLO models
        self._init_yolo()

        cap = cv2.VideoCapture(self.camera_source)
        if not cap.isOpened():
            log.error(f"Cannot open camera: {self.camera_source}")
            return

        # Warmup
        for _ in range(3):
            cap.read()

        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        log.info(f"Camera ready: {w}x{h}  |  parts: {self.parts}")

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    log.warning("Failed to read frame, retrying...")
                    time.sleep(0.05)
                    continue

                self._frame_id += 1

                # Skip frames to reduce load
                if self._frame_id % self.frame_skip != 0:
                    continue

                # Only process parts not yet captured
                pending = [p for p, s in self._states.items()
                           if not s.captured and s.retry_count < self.max_retries]

                if not pending:
                    log.info("All parts captured or max retries reached. Stopping.")
                    break

                # Get part_ids for pending parts
                pending_ids = {self._resolve_part_id(p) for p in pending if self._resolve_part_id(p)}

                # YOLO detect all pending parts in this frame
                detections = self._yolo_manager.detect_pending(frame, pending_ids)

                # Process each detection
                for det in detections:
                    part_id = det["part_id"]
                    # Reverse-map part_id → part code
                    part = self._part_id_to_code(part_id)
                    if part is None:
                        continue
                    self._process_detection(frame, part, det["bbox"])

        finally:
            cap.release()
            log.info(f"Camera released. "
                     f"Captured: {sum(1 for s in self._states.values() if s.captured)}"
                     f"/{len(self._states)}")

    # ──────────────────────────────────────────────────────────
    # Per-detection processing
    # ──────────────────────────────────────────────────────────

    def _process_detection(self,
                            frame: np.ndarray,
                            part: str,
                            yolo_bbox: List[float]):
        """
        Run overlap check + IQA for one detection.
        Saves the image and fires on_capture callback if everything passes.
        """
        state = self._states.get(part)
        if state is None or state.captured:
            return

        redframe_svc = self._get_redframe_service(part)

        # Merge position to the format RedFrameService expects
        from routers.utils import merge_position
        position_merged = merge_position(yolo_bbox)

        # forward() runs overlap check + IQA internally
        result = redframe_svc.forward(
            image=frame,
            frame=None,           # no normalized frame string needed here
            position=position_merged,
            crop_method=None,     # uses crop_method_default
        )

        # Update state for logging
        if result.iqa_result:
            state.last_iqa = result.iqa_result.iqa_score

        if not result.result:
            if result.error_type in ("wide_error", "out_of_frame_error"):
                log.debug(f"[{self.category}/{part}] "
                          f"frame={self._frame_id} "
                          f"overlap FAIL: {result.error_type}")
            elif result.error_type == "iqa_error":
                state.retry_count += 1
                log.info(f"[{self.category}/{part}] "
                         f"IQA FAIL score={state.last_iqa:.4f} "
                         f"(retry {state.retry_count}/{self.max_retries})")
            return

        # ── PASS: save + callback ──────────────────────────────
        state.captured = True

        saved_path = None
        if self.save_dir:
            fname = self.save_dir / f"{self.category}_{part}_f{self._frame_id:06d}.jpg"
            # Crop by redframe to save only the relevant region
            from .redframe_service_fix import RedFrameService
            crop = redframe_svc.crop_method_default(frame, yolo_bbox)
            cv2.imwrite(str(fname), crop)
            saved_path = str(fname)

        capture = CaptureResult(
            part_id=part,
            category=self.category,
            frame_id=self._frame_id,
            iqa_score=result.iqa_result.iqa_score,
            iqa_threshold=result.iqa_result.iqa_threshold,
            bbox=yolo_bbox,
            saved_path=saved_path,
        )

        log.info(f"CAPTURED ✓ {capture}")

        if self.on_capture:
            try:
                self.on_capture(capture)
            except Exception as e:
                log.error(f"on_capture callback error: {e}")

    # ──────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────

    def _part_id_to_code(self, part_id: str) -> Optional[str]:
        """Reverse-map part_id (rolex_p01) back to part code (P01)."""
        # part_id format: "{prefix}_{part_num}" e.g. "rolex_p01" or "ctr_p01-1"
        parts = part_id.split("_")
        if len(parts) < 2:
            return None
        raw = parts[-1].split("-")[0]     # "p01-1" → "p01"
        code = raw.upper().replace("P0", "P0")  # "p01" → "P01"
        return code if code in self._states else None

    @property
    def progress(self) -> str:
        done = sum(1 for s in self._states.values() if s.captured)
        return f"{done}/{len(self._states)}"

    @property
    def captured_parts(self) -> List[str]:
        return [p for p, s in self._states.items() if s.captured]

    @property
    def pending_parts(self) -> List[str]:
        return [p for p, s in self._states.items() if not s.captured]


# ══════════════════════════════════════════════════════════════
# FastAPI integration helpers
# ══════════════════════════════════════════════════════════════

# Global registry of running services (one per category/session)
_services: Dict[str, AutoCaptureService] = {}


def get_or_create_service(category: str,
                           parts: List[str],
                           camera_source=0,
                           device: str = "cuda",
                           on_capture: Optional[Callable] = None,
                           **kwargs) -> AutoCaptureService:
    """
    Get existing service for a category or create a new one.
    Call this from a FastAPI route or lifespan handler.
    """
    key = f"{category}_{camera_source}"
    if key not in _services:
        _services[key] = AutoCaptureService(
            category=category,
            parts=parts,
            camera_source=camera_source,
            device=device,
            on_capture=on_capture,
            **kwargs,
        )
    return _services[key]