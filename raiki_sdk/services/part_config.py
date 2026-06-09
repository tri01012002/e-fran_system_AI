"""
Part Config Registry
====================
Define part config for each class/part corresponding to the YOLO model.
Support for bbox
"""

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
    # bbox: [x1, y1, x2, y2]
    redframe: Union[List[float], List[List[float]]]
    tolerance: float = 0.15
    min_iou: float = None
    description: str = ""

    def __post_init__(self):
        if self.min_iou is None:
            self.min_iou = 1.0 - self.tolerance

# ──────────────────────────────────────────────────────────────
# SAMPLE CONFIG FOR EACH PART
# ──────────────────────────────────────────────────────────────

PART_CONFIGS: dict[str, PartConfig] = {
    # -----Cartier------
    "ctr_p01-1": PartConfig(
        part_id="ctr_p01-1",
        redframe_type="bbox",
        redframe=[47, 303, 279, 390],
        description="Cartier P01 Type1"
    ),

    "ctr_p01-2": PartConfig(
        part_id="ctr_p01-2",
        redframe_type="bbox",
        redframe=[57, 313, 271.41, 364],
        description="Cartier P01 Type2"
    ),

    "ctr_p02-1": PartConfig(
        part_id="ctr_p02-1",
        redframe_type="bbox",
        redframe=[150, 248.5, 183, 437],
        description="Cartier P02 Type1"
    ),

    "ctr_p02-2": PartConfig(
        part_id="ctr_p02-2",
        redframe_type="bbox",
        redframe=[139, 264, 180.5, 437],
        description="Cartier P02 Type2"
    ),

    "ctr_p03-1": PartConfig(
        part_id="ctr_p03-1",
        redframe_type="bbox",
        redframe=[130, 292, 190, 352],
        description="Cartier P03 Type1"
    ),

    "ctr_p03-2": PartConfig(
        part_id="ctr_p03-2",
        redframe_type="bbox",
        redframe=[130, 292, 190, 352],
        description="Cartier P03 Type2"
    ),

    "ctr_p04": PartConfig(
        part_id="ctr_p04",
        redframe_type="bbox",
        redframe=[44, 282, 276, 369],
        description="Cartier P04"
    ),

    "ctr_p05": PartConfig(
        part_id="ctr_p05",
        redframe_type="bbox",
        redframe=[44, 282, 276, 369],
        description="Cartier P05"
    ),

    "ctr_p06": PartConfig(
        part_id="ctr_p06",
        redframe_type="bbox",
        redframe=[97, 262, 224, 390],
        description="Cartier P06"
    ),

    #-----Omega-----
    "omg_p01": PartConfig(
        part_id="omg_p01",
        redframe_type="bbox",
        redframe=[67, 219, 254, 369],
        description="Omega P01"
    ),

    "omg_p02": PartConfig(
        part_id="omg_p02",
        redframe_type="bbox",
        redframe=[114, 307, 207, 436],
        description="Omega P02"
    ),

    "omg_p03": PartConfig(
        part_id="omg_p03",
        redframe_type="bbox",
        redframe=[130, 292, 190, 352],
        description="Omega P03"
    ),

    "omg_p05": PartConfig(
        part_id="omg_p05",
        redframe_type="bbox",
        redframe=[80, 242, 240, 425],
        description="Omega P05"
    ),

    "omg_p06": PartConfig(
        part_id="omg_p06",
        redframe_type="bbox",
        redframe=[60, 226, 260, 426],
        description="Omega P06"
    ),

    #-----Rolex-----
    "rolex_p01": PartConfig(
        part_id="rolex_p01",
        redframe_type="bbox",
        redframe=[90, 341, 230, 370],
        description="Rolex P01"
    ),

    "rolex_p02": PartConfig(
        part_id="rolex_p02",
        redframe_type="bbox",
        redframe=[40, 270, 280, 377],
        description="Rolex P02"
    ),

    "rolex_p03": PartConfig(
        part_id="rolex_p03",
        redframe_type="bbox",
        redframe=[130, 292, 190, 352],
        description="Rolex P03"
    ),

    "rolex_p04": PartConfig(
        part_id="rolex_p04",
        redframe_type="bbox",
        redframe=[65, 290, 255, 370],
        description="Rolex P04"
    ),

    "rolex_p05": PartConfig(
        part_id="rolex_p05",
        redframe_type="bbox",
        redframe=[127, 290, 192, 355],
        description="Rolex P05"
    ),

    "rolex_p08": PartConfig(
        part_id="rolex_p08",
        redframe_type="bbox",
        redframe=[76, 294, 244, 352],
        description="Rolex P08"
    ),

    "rolex_p10": PartConfig(
        part_id="rolex_p10",
        redframe_type="bbox",
        redframe=[14, 239, 305, 446],
        description="Rolex P10"
    )
}

def get_config(part_id: str) -> PartConfig:
    if part_id not in PART_CONFIGS:
        raise KeyError(f"Part {part_id} does not exist in config."
                        f"Part existed: {list(PART_CONFIGS.keys())}")
    return PART_CONFIGS[part_id]


def save_configs(path: str):
    """Export config to JSON file for modifying."""
    data = {}
    for pid, cfg in PART_CONFIGS.items():
        data[pid] ={
            "redframe_type": cfg.redframe_type,
            "redframe": cfg.redframe,
            "tolerance": cfg.tolerance,
            "description": cfg.description,
        }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[Config] Saved {len(data)} parts → {path}")


def load_configs_from_json(path: str):
    """Load config from file JSON."""
    global PART_CONFIGS
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    PART_CONFIGS = {}
    for pid, cfg in data.items():
        PART_CONFIGS[pid] = PartConfig(
            part_id=pid,
            redframe_type=cfg["redframe_type"],
            redframe=cfg["redframe"],
            tolerance=cfg.get("tolerance", 0.15),
            description=cfg.get("description", ""),
        )
    print(f"[Config] Loaded {len(PART_CONFIGS)} parts from {path}")