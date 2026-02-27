
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

from PIL import Image

DistanceValue = Union[float, int, str, None]


@dataclass(frozen=True)
class MatchItemModel:
    index: int

    original_image: Optional[Image.Image]
    matched_image: Optional[Image.Image]

    original_meta: Dict[str, str]
    matched_meta: Dict[str, str]

    distance: DistanceValue
    metric_name: str
