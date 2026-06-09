"""
overlap_metrics.py
==================
Compute overlap metrics between a YOLO bounding box and a redframe.

Only bounding boxes (rectangles) are supported because all parts use bbox-based redframes.

Metrics:
    iou    : Intersection over Union          [0..1]
    iom    : Intersection over Redframe Area  [0..1]  ← primary metric
    passed : bool — iom >= (1 - tolerance)

Why use IoM instead of IoU?
    A YOLO bbox may be slightly larger than the redframe while still being valid.
    IoU penalizes such cases, whereas IoM does not — it only measures
    whether the redframe is sufficiently covered.

Usage:
    from overlap_metrics import compute_overlap
    from part_config import get_config

    cfg = get_config("rolex_p01")
    metrics = compute_overlap([x1, y1, x2, y2], cfg)

    if metrics["passed"]:
        trigger_capture()
    else:
        print(f"Deviation {(1 - metrics['iom']) * 100:.1f}% exceeds tolerance")
"""

from typing import List, Dict, Any, Union
from .part_config import PartConfig


def compute_overlap(
    yolo_bbox: List[float],
    cfg: Union[PartConfig, dict]
) -> Dict[str, Any]:
    """
    Compute overlap metrics between a YOLO bounding box and a redframe.

    Args:
        yolo_bbox:
            [x1, y1, x2, y2] bounding box from YOLO
            (pixel coordinates within the frame)

        cfg:
            PartConfig instance or dict containing:
                redframe
                tolerance

            redframe_type must be "bbox".

    Returns:
        dict:
            iou:
                Intersection over Union

            iom:
                Intersection over Redframe Area (primary metric)

            tolerance:
                Tolerance value loaded from configuration

            passed:
                True if iom >= (1 - tolerance)

            overlap_pct:
                Percentage of redframe covered (= iom * 100)

            deviation_pct:
                Percentage deviation (= (1 - iom) * 100)
    """

    # Read values from PartConfig or dict
    if isinstance(cfg, dict):
        redframe = cfg["redframe"]
        tolerance = cfg.get("tolerance", 0.15)
    else:
        redframe = cfg.redframe
        tolerance = cfg.tolerance

    # ── YOLO bounding box ─────────────────────────────────────
    yx1, yy1, yx2, yy2 = yolo_bbox
    area_yolo = max(0.0, yx2 - yx1) * max(0.0, yy2 - yy1)

    # ── Redframe bounding box ─────────────────────────────────
    rx1, ry1, rx2, ry2 = redframe
    area_red = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)

    # ── Intersection area ─────────────────────────────────────
    ix1 = max(yx1, rx1)
    iy1 = max(yy1, ry1)
    ix2 = min(yx2, rx2)
    iy2 = min(yy2, ry2)

    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    # ── Metrics ───────────────────────────────────────────────
    union = area_yolo + area_red - inter

    iou = inter / union if union > 0 else 0.0
    iom = inter / area_red if area_red > 0 else 0.0

    passed = iom >= (1.0 - tolerance)

    return {
        "iou": round(iou, 4),
        "iom": round(iom, 4),
        "tolerance": tolerance,
        "passed": passed,
        "overlap_pct": round(iom * 100, 2),
        "deviation_pct": round((1 - iom) * 100, 2),
    }


def compute_overlap_batch(
    detections: List[dict],
    configs: dict
) -> List[dict]:
    """
    Compute overlap metrics for multiple detections.

    Args:
        detections:
            List of detections in the format:
            {
                "part_id": str,
                "bbox": [x1, y1, x2, y2]
            }

        configs:
            Dictionary:
            {
                part_id: PartConfig | dict
            }

    Returns:
        List of results:
        {
            "part_id": str,
            "bbox": [x1, y1, x2, y2],
            "metrics": dict,
            "passed": bool
        }
    """

    results = []

    for det in detections:
        part_id = det["part_id"]
        bbox = det["bbox"]
        cfg = configs.get(part_id)

        if cfg is None:
            results.append({
                "part_id": part_id,
                "bbox": bbox,
                "metrics": None,
                "passed": False,
                "error": f"Part '{part_id}' does not exist in the configuration",
            })
            continue

        metrics = compute_overlap(bbox, cfg)

        results.append({
            "part_id": part_id,
            "bbox": bbox,
            "metrics": metrics,
            "passed": metrics["passed"],
        })

    return results