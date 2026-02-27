
from __future__ import annotations

from typing import Any, Optional, Sequence

from .controllers.viz_controller import MatchVisualizerController
from .views.viz_view import MatchesView


def visualize_matches(items: Sequence[Any], max_size: int = 300, metric_name: Optional[str] = None) -> None:
    controller = MatchVisualizerController(view=MatchesView(title="LBP Matches"))
    controller.visualize(items, max_size=max_size, metric_name=metric_name)
