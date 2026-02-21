import tkinter as tk
from tkinter import ttk
from typing import List, Dict, Optional
from PIL import Image, ImageTk


def visualize_matches(items: List[Dict], max_size: int = 300) -> None:
    """Display a scrollable window showing each item's IMAGE and its matched IMAGE.

    Each dict in `items` must contain:
    - 'IMAGE': a PIL.Image
    - 'MATCHED_INDEX': index of the matched item in the items list (or -1/None)
    - metadata keys like 'INSTANCE' and 'CATEGORY' for labels
    """
    if not items:
        print('No items to visualize')
        return

    root = tk.Tk()
    root.title('LBP Matches')

    canvas = tk.Canvas(root)
    scrollbar = ttk.Scrollbar(root, orient='vertical', command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side='left', fill='both', expand=True)
    scrollbar.pack(side='right', fill='y')

    photo_refs = []

    for idx, it in enumerate(items):
        frame = ttk.Frame(scrollable_frame, padding=6, relief='ridge')
        frame.pack(fill='x', padx=6, pady=6)

        left_img = it.get('IMAGE')
        match_idx = it.get('MATCHED_INDEX')
        right_img = None
        if match_idx is not None and match_idx >= 0 and match_idx < len(items):
            right_img = items[match_idx].get('IMAGE')

        # prepare thumbnails
        def make_thumb(img: Optional[Image.Image]):
            if img is None:
                return None
            im = img.copy()
            im.thumbnail((max_size, max_size), Image.LANCZOS)
            return ImageTk.PhotoImage(im)

        left_photo = make_thumb(left_img)
        right_photo = make_thumb(right_img)
        photo_refs.append((left_photo, right_photo))

        left_label = ttk.Label(frame)
        if left_photo:
            left_label.configure(image=left_photo)
        else:
            left_label.configure(text='(no image)')
        left_label.grid(row=0, column=0, rowspan=2, padx=8)

        right_label = ttk.Label(frame)
        if right_photo:
            right_label.configure(image=right_photo)
        else:
            right_label.configure(text='(no match)')
        right_label.grid(row=0, column=1, rowspan=2, padx=8)

        # collect all string-valued metadata from the item and the matched item
        left_meta_lines = []
        for k, v in it.items():
            if isinstance(v, str):
                left_meta_lines.append(f"{k}: {v}")

        right_meta_lines = []
        if match_idx is not None and match_idx >= 0 and match_idx < len(items):
            matched = items[match_idx]
            for k, v in matched.items():
                if isinstance(v, str):
                    right_meta_lines.append(f"{k}: {v}")
        else:
            matched = None

        distance = it.get('DISTANCE')
        dist_line = f"Distance: {distance:.6f}" if (distance is not None and not isinstance(distance, str)) else f"Distance: {distance}"

        info_lines = []
        info_lines.append(f"Index: {idx}")
        info_lines.append("-- Original --")
        info_lines.extend(left_meta_lines)
        info_lines.append("")
        info_lines.append("-- Matched --")
        if matched is not None:
            info_lines.extend(right_meta_lines)
        else:
            info_lines.append('(no match)')
        info_lines.append("")
        info_lines.append(dist_line)

        info_text = "\n".join(info_lines)
        ttk.Label(frame, text=info_text, justify='left', anchor='w', wraplength=600).grid(row=0, column=2, sticky='w')

    root.mainloop()