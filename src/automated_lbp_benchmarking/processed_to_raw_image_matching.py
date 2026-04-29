from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence
import numpy as np
from .distance_metrics import chi2_distance, get_distance_metric
from .image_data_containers import ImageRecord, MatchRecord

@dataclass
class ProcessedToRawMatcher:
    distance_fn: Callable[[np.ndarray, np.ndarray], float] = chi2_distance
    metric_name: str = "chi2"
    tolerance: Optional[float] = 1
    top: Optional[int] = 1

    def __call__(self, processed_records: Sequence[ImageRecord], raw_records: Sequence[ImageRecord]) -> Sequence[ImageRecord]:
        if self.tolerance is None:
            self.tolerance = 1
        if self.top is None:
            self.top = 1
        if not processed_records or not raw_records:
            return processed_records

        distance_fn = self.distance_fn
        if self.metric_name:
            distance_fn = get_distance_metric(self.metric_name)

        raw_hists = [np.asarray(r.lbp_hist, dtype=np.float64) for r in raw_records]
        num_raw_records = len(raw_hists)

        for record_index, proc_record in enumerate(processed_records):
            proc_record.index = record_index
            match_records: List[MatchRecord] = []
            visited = []
            min_val = float("inf")
            min_idx = -1

            for top_n_match_index in range(self.top):
                for raw_record_index in range(num_raw_records):
                    if raw_record_index in visited:
                        continue
                    distance_between_feature_vectors = distance_fn(np.asarray(proc_record.lbp_hist, dtype=np.float64), raw_hists[raw_record_index])
                    if distance_between_feature_vectors < min_val and distance_between_feature_vectors <= self.tolerance:
                        min_val = distance_between_feature_vectors
                        min_idx = raw_record_index
                correct = (raw_records[min_idx].category == proc_record.category)
                matched_image = raw_records[min_idx].image if hasattr(raw_records[min_idx], "image") else None
                new_match = MatchRecord(
                    matched_index=min_idx,
                    matched_category=raw_records[min_idx].category,
                    nn_distance=min_val,
                    correct=correct,
                    matched_image=matched_image)
                visited.append(min_idx)
                if min_val == float("inf"):
                    break
                match_records.append(new_match)
                min_val = float("inf")
                min_idx = -1
            proc_record.match_records = match_records
        return processed_records