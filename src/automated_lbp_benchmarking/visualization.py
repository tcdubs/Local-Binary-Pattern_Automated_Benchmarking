from collections.abc import Sequence
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
from .image_data_containers import ImageRecord, MatchRecord


class ImageRecordMatchViewer:
    def __init__(
        self,
        image_records: Sequence["ImageRecord"],
        records_per_page: int = 50,
        thumbnail_size: tuple[int, int] = (180, 180),
    ):
        self.image_records = list(image_records)
        self.records_per_page = records_per_page
        self.thumbnail_size = thumbnail_size
        self.page = 0
        self.tk_images = []  # prevents images from being garbage-collected

        self.root = tk.Tk()
        self.root.title("Image Record Match Viewer")
        self.root.geometry("1400x900")

        self._build_layout()
        self._render_page()

    def _build_layout(self):
        nav_frame = ttk.Frame(self.root)
        nav_frame.pack(fill="x", padx=8, pady=8)

        self.prev_button = ttk.Button(nav_frame, text="Previous", command=self.prev_page)
        self.prev_button.pack(side="left")

        self.page_label = ttk.Label(nav_frame, text="")
        self.page_label.pack(side="left", padx=12)

        self.next_button = ttk.Button(nav_frame, text="Next", command=self.next_page)
        self.next_button.pack(side="left")

        self.canvas = tk.Canvas(self.root)
        self.scrollbar = ttk.Scrollbar(
            self.root,
            orient="vertical",
            command=self.canvas.yview,
        )

        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda event: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            ),
        )

        self.canvas_window = self.canvas.create_window(
            (0, 0),
            window=self.scrollable_frame,
            anchor="nw",
        )

        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _make_thumbnail(self, pil_image: Image.Image) -> ImageTk.PhotoImage:
        img = pil_image.copy()
        img.thumbnail(self.thumbnail_size)
        tk_img = ImageTk.PhotoImage(img)
        self.tk_images.append(tk_img)
        return tk_img

    def _render_page(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        self.tk_images.clear()

        start = self.page * self.records_per_page
        end = start + self.records_per_page
        page_records = self.image_records[start:end]

        total_pages = max(
            1,
            (len(self.image_records) + self.records_per_page - 1)
            // self.records_per_page,
        )

        self.page_label.config(
            text=f"Page {self.page + 1} of {total_pages}"
        )

        self.prev_button.config(state="normal" if self.page > 0 else "disabled")
        self.next_button.config(
            state="normal" if self.page < total_pages - 1 else "disabled"
        )

        for row_index, image_record in enumerate(page_records):
            self._add_image_record_row(row_index, image_record)

        self.canvas.yview_moveto(0)

    def _add_image_record_row(self, row_index: int, image_record: "ImageRecord"):
        outer = ttk.Frame(self.scrollable_frame, padding=10, relief="solid")
        outer.grid(row=row_index, column=0, sticky="ew", padx=8, pady=8)

        # Left: ImageRecord
        image_frame = ttk.Frame(outer, padding=8, relief="solid")
        image_frame.grid(row=0, column=0, sticky="n", padx=(0, 12))

        main_img = self._make_thumbnail(image_record.image)
        img_label = ttk.Label(image_frame, image=main_img)
        img_label.pack()

        metadata = (
            f"Category: {image_record.category}\n"
            f"Rotation: {image_record.rotation}\n"
            f"Distance: {image_record.distance}\n"
            f"Lighting: {image_record.lighting}"
        )

        ttk.Label(
            image_frame,
            text=metadata,
            justify="left",
            font=("Consolas", 10),
        ).pack(anchor="w", pady=(8, 0))

        # Right: MatchRecords
        matches_frame = ttk.Frame(outer)
        matches_frame.grid(row=0, column=1, sticky="nw")

        for match_index, match_record in enumerate(image_record.match_records):
            self._add_match_card(matches_frame, match_index, match_record)

    def _add_match_card(
        self,
        parent: ttk.Frame,
        match_index: int,
        match_record: "MatchRecord",
    ):
        # ttk widgets do not reliably support custom background colors,
        # so this match card uses tk.Frame/tk.Label instead.
        if match_record.correct is True:
            bg_color = "#e6f7e6"  # light green
        elif match_record.correct is False:
            bg_color = "#fbeaea"  # light red
        else:
            bg_color = "#f0f0f0"  # neutral gray for unknown

        card = tk.Frame(
            parent,
            bd=1,
            relief="solid",
            bg=bg_color,
            padx=8,
            pady=8,
        )
        card.grid(row=0, column=match_index, padx=6, sticky="n")

        tk.Label(
            card,
            text=f"MatchRecord[{match_index}]",
            font=("Consolas", 10, "bold"),
            bg=bg_color,
            anchor="w",
        ).pack(anchor="w")

        if match_record.matched_image is not None:
            match_img = self._make_thumbnail(match_record.matched_image)
            tk.Label(card, image=match_img, bg=bg_color).pack(pady=(4, 8))
        else:
            tk.Label(
                card,
                text="[No image]",
                width=24,
                anchor="center",
                bg=bg_color,
            ).pack(pady=(4, 8))

        distance_text = (
            f"{match_record.nn_distance:.6f}"
            if match_record.nn_distance is not None
            else "None"
        )

        info = (
            f"Category: {match_record.matched_category}\n"
            f"Distance: {distance_text}\n"
            f"Correct: {match_record.correct}"
        )

        tk.Label(
            card,
            text=info,
            justify="left",
            font=("Consolas", 10),
            bg=bg_color,
            anchor="w",
        ).pack(anchor="w")

    def next_page(self):
        max_page = (
            len(self.image_records) + self.records_per_page - 1
        ) // self.records_per_page - 1

        if self.page < max_page:
            self.page += 1
            self._render_page()

    def prev_page(self):
        if self.page > 0:
            self.page -= 1
            self._render_page()

    def run(self):
        self.root.mainloop()


def visualize_image_records(
    image_records: Sequence["ImageRecord"],
    records_per_page: int = 50,
    thumbnail_size: tuple[int, int] = (180, 180),
) -> None:
    viewer = ImageRecordMatchViewer(
        image_records=image_records,
        records_per_page=records_per_page,
        thumbnail_size=thumbnail_size,
    )
    viewer.run()
