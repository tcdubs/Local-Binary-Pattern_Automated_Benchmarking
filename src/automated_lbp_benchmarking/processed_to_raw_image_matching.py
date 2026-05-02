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
            raise ValueError("No records provided for matching.")

        # Get distance function based on metric name
        distance_fn = self.distance_fn
        if self.metric_name:
            distance_fn = get_distance_metric(self.metric_name)

        # Get all LBP histograms from raw 'target' records
        raw_hists = [np.asarray(r.lbp_hist, dtype=np.float64) for r in raw_records]

        # Loop through each processed (query) record and compute distances to all raw (target) records
        # Top-k matches within tolerance parameter is stored in the processed record's match_records attribute
        for record_index, proc_record in enumerate(processed_records):
            proc_record.index = record_index
            proc_hist = np.asarray(proc_record.lbp_hist, dtype=np.float64)

            distances = np.asarray([
                distance_fn(proc_hist, raw_hist)
                for raw_hist in raw_hists
            ])
            try:
                distances = np.delete(distances, record_index)
            except:
                pass
            valid_indices = np.where(distances <= self.tolerance)[0]

            # Prevent false positive matching of final image when no matches are found/tolerance too low
            if len(valid_indices) == 0:
                proc_record.match_records = []
                continue

            top_indices = valid_indices[np.argsort(distances[valid_indices])[:self.top]]

            # Store match records list in the processed record's match_records attribute
            proc_record.match_records = [
                MatchRecord(
                    matched_index=i,
                    matched_category=raw_records[i].category,
                    nn_distance=float(distances[i]),
                    correct=(raw_records[i].category == proc_record.category),
                    matched_image=getattr(raw_records[i], "image", None),
                )
                for i in top_indices
            ]
        return processed_records