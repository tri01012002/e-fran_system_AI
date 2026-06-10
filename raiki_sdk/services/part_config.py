import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Union, Literal
import json
import os

RedFrameType = Literal["bbox"]

@dataclass
class PartConfig:
    part_id: str
    redframe_type: RedFrameType
    redframe: Union[List[float], List[List[float]]]
    tolerance: float = 0.15
    min_iou: float = None
    description: str = ""

    def __post_init__(self):
        if self.min_iou is None:
            self.min_iou = 1.0 - self.tolerance

PART_CONFIGS: dict[str, PartConfig] = {
    "ctr_p01-1": PartConfig(part_id="ctr_p01-1", redframe_type="bbox", redframe=[47, 303, 279, 390], description="Cartier P01 Type1"),
    "ctr_p01-2": PartConfig(part_id="ctr_p01-2", redframe_type="bbox", redframe=[57, 313, 271.41, 364], description="Cartier P01 Type2"),
    "ctr_p02-1": PartConfig(part_id="ctr_p02-1", redframe_type="bbox", redframe=[150, 248.5, 183, 437], description="Cartier P02 Type1"),
    "ctr_p02-2": PartConfig(part_id="ctr_p02-2", redframe_type="bbox", redframe=[139, 264, 180.5, 437], description="Cartier P02 Type2"),
    "ctr_p03-1": PartConfig(part_id="ctr_p03-1", redframe_type="bbox", redframe=[130, 292, 190, 352], description="Cartier P03 Type1"),
    "ctr_p03-2": PartConfig(part_id="ctr_p03-2", redframe_type="bbox", redframe=[130, 292, 190, 352], description="Cartier P03 Type2"),
    "ctr_p04":   PartConfig(part_id="ctr_p04",   redframe_type="bbox", redframe=[44, 282, 276, 369], description="Cartier P04"),
    "ctr_p05":   PartConfig(part_id="ctr_p05",   redframe_type="bbox", redframe=[44, 282, 276, 369], description="Cartier P05"),
    "ctr_p06":   PartConfig(part_id="ctr_p06",   redframe_type="bbox", redframe=[97, 262, 224, 390], description="Cartier P06"),
    "omg_p01":   PartConfig(part_id="omg_p01",   redframe_type="bbox", redframe=[67, 219, 254, 369], description="Omega P01"),
    "omg_p02":   PartConfig(part_id="omg_p02",   redframe_type="bbox", redframe=[114, 307, 207, 436], description="Omega P02"),
    "omg_p03":   PartConfig(part_id="omg_p03",   redframe_type="bbox", redframe=[130, 292, 190, 352], description="Omega P03"),
    "omg_p05":   PartConfig(part_id="omg_p05",   redframe_type="bbox", redframe=[80, 242, 240, 425], description="Omega P05"),
    "omg_p06":   PartConfig(part_id="omg_p06",   redframe_type="bbox", redframe=[60, 226, 260, 426], description="Omega P06"),
    "rolex_p01": PartConfig(part_id="rolex_p01", redframe_type="bbox", redframe=[90, 341, 230, 370], description="Rolex P01"),
    "rolex_p02": PartConfig(part_id="rolex_p02", redframe_type="bbox", redframe=[40, 270, 280, 377], description="Rolex P02"),
    "rolex_p03": PartConfig(part_id="rolex_p03", redframe_type="bbox", redframe=[130, 292, 190, 252], description="Rolex P03"),
    "rolex_p04": PartConfig(part_id="rolex_p04", redframe_type="bbox", redframe=[65, 290, 255, 370], description="Rolex P04"),
    "rolex_p05": PartConfig(part_id="rolex_p05", redframe_type="bbox", redframe=[127, 290, 192, 355], description="Rolex P05"),
    "rolex_p08": PartConfig(part_id="rolex_p08", redframe_type="bbox", redframe=[76, 294, 244, 352], description="Rolex P08"),
    "rolex_p10": PartConfig(part_id="rolex_p10", redframe_type="bbox", redframe=[14, 239, 305, 446], description="Rolex P10"),
}

def get_config(part_id: str) -> PartConfig:
    if part_id not in PART_CONFIGS:
        raise KeyError(f"Part {part_id} does not exist. Available: {list(PART_CONFIGS.keys())}")
    return PART_CONFIGS[part_id]

def save_configs(path: str):
    data = {pid: {"redframe_type": cfg.redframe_type, "redframe": cfg.redframe,
                  "tolerance": cfg.tolerance, "description": cfg.description}
            for pid, cfg in PART_CONFIGS.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def load_configs_from_json(path: str):
    global PART_CONFIGS
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    PART_CONFIGS = {pid: PartConfig(part_id=pid, redframe_type=cfg["redframe_type"],
                    redframe=cfg["redframe"], tolerance=cfg.get("tolerance", 0.15),
                    description=cfg.get("description", ""))
                   for pid, cfg in data.items()}


# ══════════════════════════════════════════════════════════════
# SCALE UTILITIES
# ══════════════════════════════════════════════════════════════

# Figma absolute frame size (check in Figma: frame W x H)
FIGMA_W: int = 320
FIGMA_H: int = 667

# Captured image size from camera
IMAGE_W: int = 2000
IMAGE_H: int = 1394


def scale_redframe(
    bbox: List[float],
    figma_w: int = FIGMA_W,
    figma_h: int = FIGMA_H,
    image_w: int = IMAGE_W,
    image_h: int = IMAGE_H,
    mode: str = "stretch",
) -> List[float]:
    """
    Scale a redframe bbox from Figma space to image space.

    Args:
        bbox    : [x1, y1, x2, y2] in Figma coordinates
        figma_w : Figma frame width  (default: FIGMA_W)
        figma_h : Figma frame height (default: FIGMA_H)
        image_w : captured image width  (default: IMAGE_W)
        image_h : captured image height (default: IMAGE_H)
        mode    : one of:
                  "stretch"    — scale X and Y independently (no ratio preserved)
                  "fit_width"  — preserve ratio, fit to width, pad top/bottom
                  "fit_height" — preserve ratio, fit to height, pad left/right

    Returns:
        [x1, y1, x2, y2] in image pixel coordinates
    """
    x1, y1, x2, y2 = bbox

    if mode == "stretch":
        sx = image_w / figma_w
        sy = image_h / figma_h
        return [x1*sx, y1*sy, x2*sx, y2*sy]

    elif mode == "fit_width":
        # Scale so Figma width fills image width; letterbox top/bottom
        s = image_w / figma_w
        rendered_h = figma_h * s
        pad_y = (image_h - rendered_h) / 2
        return [x1*s, y1*s + pad_y, x2*s, y2*s + pad_y]

    elif mode == "fit_height":
        # Scale so Figma height fills image height; letterbox left/right
        s = image_h / figma_h
        rendered_w = figma_w * s
        pad_x = (image_w - rendered_w) / 2
        return [x1*s + pad_x, y1*s, x2*s + pad_x, y2*s]

    else:
        raise ValueError(f"Unknown scale mode: '{mode}'. "
                         f"Use 'stretch', 'fit_width', or 'fit_height'.")


def build_scaled_configs(
    mode: str = "stretch",
    figma_w: int = FIGMA_W,
    figma_h: int = FIGMA_H,
    image_w: int = IMAGE_W,
    image_h: int = IMAGE_H,
) -> dict:
    """
    Return a new dict of PartConfigs with all redframes scaled
    from Figma space to image space.

    Call this once at startup and use the result throughout inference:
        SCALED_CONFIGS = build_scaled_configs(mode="stretch")

    Args:
        mode    : "stretch" | "fit_width" | "fit_height"
        figma_w : Figma frame width
        figma_h : Figma frame height
        image_w : captured image width
        image_h : captured image height

    Returns:
        dict {part_id: PartConfig} with scaled redframe coordinates
    """
    scaled = {}
    for part_id, cfg in PART_CONFIGS.items():
        scaled_bbox = scale_redframe(
            cfg.redframe, figma_w, figma_h, image_w, image_h, mode
        )
        scaled[part_id] = PartConfig(
            part_id=cfg.part_id,
            redframe_type=cfg.redframe_type,
            redframe=scaled_bbox,
            tolerance=cfg.tolerance,
            min_iou=cfg.min_iou,
            description=cfg.description,
        )
    return scaled