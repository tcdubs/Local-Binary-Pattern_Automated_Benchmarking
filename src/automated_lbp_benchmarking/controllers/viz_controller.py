from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from PIL import Image, ImageGrab

from ..models.viz_model import MatchItemModel
from ..views.viz_view import MatchesView


def _pretty_metric_name(metric: Optional[str]) -> str:
    if not metric:
        return "Chi Squared"
    m = metric.strip().lower()
    if m in ("chi2", "chisq", "chi-square", "chi squared", "chi-squared"):
        return "Chi Squared"
    if m == "cosine":
        return "Cosine"
    if m == "hellinger":
        return "Hellinger"
    return metric.strip().title()


def _is_imagerecord_like(obj: Any) -> bool:
    return hasattr(obj, "image") and hasattr(obj, "lbp_hist") and hasattr(obj, "category")


def _extract_string_meta_from_mapping(m: Mapping[str, Any]) -> Dict[str, str]:
    return {k: v for k, v in m.items() if isinstance(v, str)}


def _extract_string_meta_from_obj(obj: Any) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for attr, label in [
        ("instance", "INSTANCE"),
        ("category", "CATEGORY"),
        ("distance", "DISTANCE"),
        ("rotation", "ROTATION"),
        ("lighting", "LIGHTING"),
    ]:
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if val is not None:
                out[label] = str(val)
    return out


def _extract_distance_from_mapping(m: Mapping[str, Any]) -> Any:
    return m.get("DISTANCE", m.get("nn_distance", m.get("NN_DISTANCE")))


def _extract_metric_from_mapping(m: Mapping[str, Any]) -> Optional[str]:
    return (
        m.get("METRIC")
        or m.get("METRIC_NAME")
        or m.get("DISTANCE_METRIC")
        or m.get("distance_metric")
    )


@dataclass
class MatchVisualizerController:
    view: MatchesView

    def build_models(self, items: Sequence[Any], metric_name: Optional[str] = None) -> List[MatchItemModel]:
        if not items:
            return []

        models: List[MatchItemModel] = []
        default_metric_display = _pretty_metric_name(metric_name)

        for idx, it in enumerate(items):
            if isinstance(it, dict):
                original_image = it.get("IMAGE")
                matched_image = it.get("MATCHED_IMAGE")
                matched_indices = it.get("MATCHED_INDICES") or it.get("matching_indices")
                matched_distances = it.get("MATCHED_DISTANCES") or it.get("matching_distances")

                matched_images: List[Image.Image] = []
                matched_meta_list: List[Dict[str, str]] = []
                matched_distance_list: List[Any] = []

                if isinstance(matched_indices, list):
                    for match_i, match_idx in enumerate(matched_indices):
                        if not isinstance(match_idx, int) or match_idx < 0 or match_idx >= len(items):
                            continue
                        match_item = items[match_idx]
                        if isinstance(match_item, dict):
                            image = match_item.get("MATCHED_IMAGE") or match_item.get("IMAGE")
                            if isinstance(image, Image.Image):
                                matched_images.append(image)
                            matched_meta_list.append(_extract_string_meta_from_mapping(match_item))
                            if isinstance(matched_distances, list) and match_i < len(matched_distances):
                                matched_distance_list.append(matched_distances[match_i])
                            else:
                                matched_distance_list.append(_extract_distance_from_mapping(match_item))

                original_meta = _extract_string_meta_from_mapping(it)
                matched_meta = {}  # No longer using matched_item for meta

                dist = _extract_distance_from_mapping(it)
                per_item_metric = _extract_metric_from_mapping(it)
                display_metric = _pretty_metric_name(metric_name or per_item_metric)

                models.append(
                    MatchItemModel(
                        index=idx,
                        original_image=original_image if isinstance(original_image, Image.Image) else None,
                        matched_image=matched_image if isinstance(matched_image, Image.Image) else None,
                        matched_images=matched_images or None,
                        original_meta=original_meta,
                        matched_meta=matched_meta,
                        distance=dist,
                        metric_name=display_metric,
                        matched_meta_list=matched_meta_list or None,
                        matched_distances=matched_distance_list or None,
                    )
                )
                continue

            if _is_imagerecord_like(it):
                original_image = getattr(it, "image", None)

                matched_idx = getattr(it, "matched_index", None)
                matched_indices = getattr(it, "matching_indices", None)
                matched_distances = getattr(it, "matching_distances", None)

                if isinstance(matched_idx, int) and (matched_idx < 0 or matched_idx >= len(items)):
                    matched_idx = None

                matched_item = items[matched_idx] if isinstance(matched_idx, int) else None
                matched_image = getattr(matched_item, "image", None) if matched_item is not None else None

                matched_images: List[Image.Image] = []
                matched_meta_list: List[Dict[str, str]] = []
                matched_distance_list: List[Any] = []

                if isinstance(matched_indices, list):
                    for match_i, match_idx in enumerate(matched_indices):
                        if not isinstance(match_idx, int) or match_idx < 0 or match_idx >= len(items):
                            continue
                        match_item = items[match_idx]
                        image = getattr(match_item, "image", None)
                        if isinstance(image, Image.Image):
                            matched_images.append(image)
                        matched_meta_list.append(_extract_string_meta_from_obj(match_item))
                        if isinstance(matched_distances, list) and match_i < len(matched_distances):
                            matched_distance_list.append(matched_distances[match_i])
                        else:
                            matched_distance_list.append(getattr(match_item, "nn_distance", None))

                original_meta = _extract_string_meta_from_obj(it)
                matched_meta = _extract_string_meta_from_obj(matched_item) if matched_item is not None else {}

                dist = getattr(it, "nn_distance", None)

                models.append(
                    MatchItemModel(
                        index=idx,
                        original_image=original_image if isinstance(original_image, Image.Image) else None,
                        matched_image=matched_image if isinstance(matched_image, Image.Image) else None,
                        matched_images=matched_images or None,
                        original_meta=original_meta,
                        matched_meta=matched_meta,
                        distance=dist,
                        metric_name=default_metric_display,
                    )
                )
                continue

        return models

    def visualize(self, items: Sequence[Any], max_size: int = 300, metric_name: Optional[str] = None, save_pdf: bool = True, args: dict = None, summary_lines: list = None) -> None:
        import os
        from datetime import datetime
        from pathlib import Path

        models = self.build_models(items, metric_name=metric_name)
        # When tolerance mode is active, hide records that had no matches within the threshold
        if args and args.get("tolerance") is not None:
            models = [m for m in models if m.matched_images is not None and len(m.matched_images) > 0]
        self.view.show(models, max_size=max_size)

        if save_pdf:
            # Save PDF to results/ with timestamp
            project_root = Path(__file__).resolve().parents[3]
            results_dir = project_root / "results"
            results_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_path = results_dir / f"matches_{timestamp}.pdf"
            self.view.save_as_pdf(models, str(pdf_path), max_size=max_size, args=args, summary_lines=summary_lines)
            print(f"Saved visualization PDF to {pdf_path}")
