
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Iterable, Optional

from PIL import Image, ImageTk

from ..models.viz_model import MatchItemModel



class MatchesView:
    def __init__(self, title: str = "LBP Matches") -> None:
        self._title = title

    def save_as_pdf(self, models: Iterable[MatchItemModel], pdf_path: str, max_size: int = 300, args: dict = None, summary_lines: list = None) -> None:
        """
        Render a summary and all visual match rows into a compact paginated PDF.
        In multi-match mode, cap displayed matched thumbnails per record to 5.
        """
        from PIL import ImageDraw, ImageFont

        models = list(models)
        if not models:
            return

        try:
            font = ImageFont.truetype("arial.ttf", 16)
            font_small = ImageFont.truetype("arial.ttf", 12)
            font_title = ImageFont.truetype("arial.ttf", 20)
        except Exception:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()
            font_title = ImageFont.load_default()

        page_width = 1654
        page_height = 2339
        pad = 36
        section_gap = 24
        image_box = min(max_size, 220)
        matches_per_row = 3
        max_pdf_matches = 5
        line_spacing = 6
        pages = []

        def line_height_for(text_font):
            bbox = text_font.getbbox("Ag")
            return (bbox[3] - bbox[1]) + line_spacing

        def wrap_text(draw_obj, text, text_font, max_text_width):
            if not text:
                return [""]
            words = str(text).split()
            if not words:
                return [""]
            lines = []
            current = words[0]
            for word in words[1:]:
                candidate = f"{current} {word}"
                if draw_obj.textlength(candidate, font=text_font) <= max_text_width:
                    current = candidate
                else:
                    lines.append(current)
                    current = word
            lines.append(current)
            return lines

        def append_wrapped(lines, draw_obj, text, text_font, max_text_width):
            lines.extend(wrap_text(draw_obj, text, text_font, max_text_width))

        def make_thumb(img):
            if img is None:
                return None
            thumb = img.copy()
            thumb.thumbnail((image_box, image_box), Image.LANCZOS)
            if thumb.mode != "RGB":
                thumb = thumb.convert("RGB")
            return thumb

        def build_single_match_lines(model):
            info_lines = [f"Index: {model.index}", "-- Original --"]
            for k, v in model.original_meta.items():
                info_lines.append(f"{k}: {v}")
            info_lines.append("")
            info_lines.append("-- Matched --")
            if model.matched_meta:
                for k, v in model.matched_meta.items():
                    info_lines.append(f"{k}: {v}")
            else:
                info_lines.append("(no match)")
            info_lines.append("")
            info_lines.append(f"Distance Metric: {model.metric_name}")
            if model.distance is None:
                info_lines.append("Distance: None")
            elif isinstance(model.distance, (int, float)):
                info_lines.append(f"Distance: {float(model.distance):.6f}")
            else:
                info_lines.append(f"Distance: {model.distance}")
            return info_lines

        def build_multi_match_lines(model):
            info_lines = [f"Index: {model.index}"]
            if model.original_meta:
                original_meta_text = ", ".join(f"{k}={v}" for k, v in model.original_meta.items())
                info_lines.append(f"Original: {original_meta_text}")
            matched_meta_list = model.matched_meta_list or []
            total_matches = len(model.matched_images or [])
            shown_matches = min(total_matches, max_pdf_matches)
            info_lines.append(f"Shown matches: {shown_matches}/{total_matches}")
            if matched_meta_list:
                for idx, meta in enumerate(matched_meta_list[:shown_matches], start=1):
                    meta_text = ", ".join(f"{k}={v}" for k, v in meta.items())
                    if model.matched_distances and idx - 1 < len(model.matched_distances):
                        dist_val = model.matched_distances[idx - 1]
                        if isinstance(dist_val, (int, float)):
                            meta_text = f"{meta_text} | distance={float(dist_val):.6f}"
                        else:
                            meta_text = f"{meta_text} | distance={dist_val}"
                    info_lines.append(f"Match {idx}: {meta_text}")
                if total_matches > shown_matches:
                    info_lines.append(f"... {total_matches - shown_matches} more matches not shown in PDF")
            else:
                info_lines.append("(no matches)")
            info_lines.append(f"Distance Metric: {model.metric_name}")
            return info_lines

        summary_block = ["LBP Match Visualization"]
        if args:
            summary_block.append("Arguments used:")
            for k, v in args.items():
                summary_block.append(f"  {k}: {v}")
        if summary_lines:
            summary_block.extend(summary_lines)

        def new_page():
            page = Image.new("RGB", (page_width, page_height), "white")
            return page, ImageDraw.Draw(page), pad

        current_page, current_draw, current_y = new_page()

        summary_max_width = page_width - 2 * pad
        summary_line_height = line_height_for(font_small)
        summary_title_height = line_height_for(font_title)
        summary_wrapped = []
        for idx, line in enumerate(summary_block):
            if idx == 0:
                summary_wrapped.extend(wrap_text(current_draw, line, font_title, summary_max_width))
            else:
                append_wrapped(summary_wrapped, current_draw, line, font_small, summary_max_width)
        summary_height = pad + summary_title_height + max(0, (len(summary_wrapped) - 1) * summary_line_height) + section_gap

        if current_y + summary_height > page_height - pad:
            pages.append(current_page)
            current_page, current_draw, current_y = new_page()

        for idx, line in enumerate(summary_wrapped):
            text_font = font_title if idx == 0 else font_small
            current_draw.text((pad, current_y), line, fill="black", font=text_font)
            current_y += summary_title_height if idx == 0 else summary_line_height
        current_y += section_gap
        current_draw.line((pad, current_y, page_width - pad, current_y), fill="black", width=2)
        current_y += section_gap

        info_width = 520
        for model in models:
            matched_images = model.matched_images if model.matched_images is not None else ([model.matched_image] if model.matched_image else [])
            has_multi_match = model.matched_images is not None and len(model.matched_images) > 0

            if has_multi_match:
                limited_matches = matched_images[:max_pdf_matches]
                thumbs = [make_thumb(img) for img in limited_matches]
                rows = max(1, (len(thumbs) + matches_per_row - 1) // matches_per_row)
                grid_height = rows * (image_box + summary_line_height + 12)
                info_lines_raw = build_multi_match_lines(model)
            else:
                thumbs = [make_thumb(model.original_image), make_thumb(model.matched_image)]
                grid_height = image_box
                info_lines_raw = build_single_match_lines(model)

            wrapped_info_lines = []
            for line in info_lines_raw:
                append_wrapped(wrapped_info_lines, current_draw, line, font_small, info_width)
            info_height = max(summary_line_height, len(wrapped_info_lines) * summary_line_height)
            row_height = max(grid_height, info_height) + section_gap

            if current_y + row_height > page_height - pad:
                pages.append(current_page)
                current_page, current_draw, current_y = new_page()

            current_draw.rectangle((pad, current_y, page_width - pad, current_y + row_height - 10), outline="#BDBDBD", width=2)

            if has_multi_match:
                original_thumb = make_thumb(model.original_image)
                left_x = pad + 12
                top_y = current_y + 12
                if original_thumb is not None:
                    current_page.paste(original_thumb, (left_x, top_y))
                    current_draw.rectangle((left_x, top_y, left_x + image_box, top_y + image_box), outline="#444444", width=1)
                else:
                    current_draw.rectangle((left_x, top_y, left_x + image_box, top_y + image_box), outline="#444444", width=1)
                    current_draw.text((left_x + 12, top_y + image_box // 2), "(no image)", fill="black", font=font_small)

                grid_x = left_x + image_box + 24
                for idx, thumb in enumerate(thumbs):
                    thumb_x = grid_x + (idx % matches_per_row) * (image_box + 18)
                    thumb_y = top_y + (idx // matches_per_row) * (image_box + summary_line_height + 12)
                    if thumb is not None:
                        current_page.paste(thumb, (thumb_x, thumb_y))
                    current_draw.rectangle((thumb_x, thumb_y, thumb_x + image_box, thumb_y + image_box), outline="#444444", width=1)
                    dist_text = "(no distance)"
                    if model.matched_distances and idx < len(model.matched_distances):
                        dist_val = model.matched_distances[idx]
                        if isinstance(dist_val, (int, float)):
                            dist_text = f"{float(dist_val):.6f}"
                        else:
                            dist_text = str(dist_val)
                    current_draw.text((thumb_x, thumb_y + image_box + 4), dist_text, fill="black", font=font_small)
            else:
                top_y = current_y + 12
                left_x = pad + 12
                right_x = left_x + image_box + 24
                for thumb, thumb_x in zip(thumbs, [left_x, right_x]):
                    if thumb is not None:
                        current_page.paste(thumb, (thumb_x, top_y))
                    current_draw.rectangle((thumb_x, top_y, thumb_x + image_box, top_y + image_box), outline="#444444", width=1)
                    if thumb is None:
                        label = "(no image)" if thumb_x == left_x else "(no match)"
                        current_draw.text((thumb_x + 12, top_y + image_box // 2), label, fill="black", font=font_small)

            info_x = page_width - pad - info_width
            info_y = current_y + 12
            for line in wrapped_info_lines:
                current_draw.text((info_x, info_y), line, fill="black", font=font_small)
                info_y += summary_line_height

            current_y += row_height

        pages.append(current_page)

        try:
            rgb_pages = [page.convert("RGB") for page in pages]
            rgb_pages[0].save(pdf_path, "PDF", resolution=100.0, save_all=True, append_images=rgb_pages[1:])
        except Exception as e:
            print(f"Warning: Could not save PDF: {e}")

    def show(self, models: Iterable[MatchItemModel], max_size: int = 300) -> None:
        models = list(models)  # No cap for runtime visualization
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
            matched_images = m.matched_images if m.matched_images is not None else ([m.matched_image] if m.matched_image else [])
            has_tolerance_matches = m.matched_images is not None and len(m.matched_images) > 0
            
            frame = ttk.Frame(scrollable_frame, padding=6, relief="ridge")
            frame.pack(fill="x", padx=6, pady=6)

            if has_tolerance_matches:
                # Multi-match layout (tolerance mode): show in grid with wrapping
                left_photo = make_thumb(m.original_image)
                photo_refs.append(left_photo)

                left_label = ttk.Label(frame)
                if left_photo:
                    left_label.configure(image=left_photo)
                else:
                    left_label.configure(text="(no image)")
                left_label.grid(row=0, column=0, padx=8, pady=4, rowspan=2)

                # Original metadata
                orig_meta_text = "\n".join(f"{k}: {v}" for k, v in m.original_meta.items())
                ttk.Label(frame, text=orig_meta_text, justify="left", anchor="w").grid(
                    row=0, column=1, sticky="w", padx=4
                )

                # Matched images in a grid with wrapping (max 3 per row)
                matched_frame = ttk.Frame(frame)
                matched_frame.grid(row=1, column=1, sticky="w", padx=4, pady=4)
                
                images_per_row = 3
                for idx, match_img in enumerate(matched_images):
                    match_thumb_photo = make_thumb(match_img) if isinstance(match_img, Image.Image) else None
                    photo_refs.append(match_thumb_photo)
                    
                    row_idx = idx // images_per_row
                    col_idx = idx % images_per_row
                    
                    col_frame = ttk.Frame(matched_frame)
                    col_frame.grid(row=row_idx, column=col_idx, padx=2, pady=2)
                    
                    match_label = ttk.Label(col_frame)
                    if match_thumb_photo:
                        match_label.configure(image=match_thumb_photo)
                    else:
                        match_label.configure(text="(no image)")
                    match_label.pack()
                    
                    # Distance underneath
                    dist_str = "(no distance)"
                    if m.matched_distances and idx < len(m.matched_distances):
                        dist_val = m.matched_distances[idx]
                        if isinstance(dist_val, (int, float)):
                            dist_str = f"{float(dist_val):.6f}"
                        else:
                            dist_str = str(dist_val)
                    ttk.Label(col_frame, text=dist_str, font=("TkDefaultFont", 9)).pack()
            else:
                # Single match layout (original mode): side by side
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
        
