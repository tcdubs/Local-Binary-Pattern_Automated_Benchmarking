
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Iterable, Optional

from PIL import Image, ImageTk

from ..models.viz_model import MatchItemModel


class MatchesView:
    def __init__(self, title: str = "LBP Matches") -> None:
        self._title = title

    def show(self, models: Iterable[MatchItemModel], max_size: int = 300) -> None:
        models = list(models)
        if not models:
            print("No items to visualize")
            return

        root = tk.Tk()
        root.title(self._title)

        canvas = tk.Canvas(root)
        scrollbar = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Keep PhotoImage references alive for the duration of the UI
        photo_refs = []

        def make_thumb(img: Optional[Image.Image]) -> Optional[ImageTk.PhotoImage]:
            if img is None:
                return None
            im = img.copy()
            im.thumbnail((max_size, max_size), Image.LANCZOS)
            return ImageTk.PhotoImage(im)

        for m in models:
            frame = ttk.Frame(scrollable_frame, padding=6, relief="ridge")
            frame.pack(fill="x", padx=6, pady=6)

            left_photo = make_thumb(m.original_image)
            right_photo = make_thumb(m.matched_image)
            photo_refs.append((left_photo, right_photo))

            left_label = ttk.Label(frame)
            if left_photo:
                left_label.configure(image=left_photo)
            else:
                left_label.configure(text="(no image)")
            left_label.grid(row=0, column=0, rowspan=2, padx=8)

            right_label = ttk.Label(frame)
            if right_photo:
                right_label.configure(image=right_photo)
            else:
                right_label.configure(text="(no match)")
            right_label.grid(row=0, column=1, rowspan=2, padx=8)

            info_lines = []
            info_lines.append(f"Index: {m.index}")
            info_lines.append("-- Original --")
            for k, v in m.original_meta.items():
                info_lines.append(f"{k}: {v}")
            info_lines.append("")
            info_lines.append("-- Matched --")
            if m.matched_meta:
                for k, v in m.matched_meta.items():
                    info_lines.append(f"{k}: {v}")
            else:
                info_lines.append("(no match)")
            info_lines.append("")
            info_lines.append(f"Distance Metric: {m.metric_name}")

            if m.distance is None:
                dist_line = "Distance: None"
            elif isinstance(m.distance, (int, float)):
                dist_line = f"Distance: {float(m.distance):.6f}"
            else:
                dist_line = f"Distance: {m.distance}"
            info_lines.append(dist_line)

            info_text = "\n".join(info_lines)
            ttk.Label(frame, text=info_text, justify="left", anchor="w", wraplength=600).grid(
                row=0, column=2, sticky="w"
            )

        root.mainloop()
