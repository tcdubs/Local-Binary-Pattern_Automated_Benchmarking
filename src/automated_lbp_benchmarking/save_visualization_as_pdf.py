from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO
from math import ceil
from pathlib import Path
from typing import Optional

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.lib.enums import TA_LEFT

from .image_data_containers import ImageRecord, MatchRecord


def pil_to_rl_image(
    pil_img: Image.Image,
    max_size: tuple[int, int] = (90, 90),
) -> RLImage:
    """Convert a PIL image into a ReportLab image while preserving aspect ratio."""
    img = pil_img.copy()

    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    img.thumbnail(max_size)

    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)

    rl_img = RLImage(buffer, width=img.width, height=img.height)
    return rl_img


def _distance_text(distance: Optional[float]) -> str:
    if distance is None:
        return "None"
    return f"{distance:.6f}"


def _match_background(correct: Optional[bool]):
    if correct is True:
        return colors.Color(0.70, 0.95, 0.70)  # light green
    if correct is False:
        return colors.Color(0.98, 0.55, 0.55)  # light red
    return colors.lightgrey


def _build_main_record_card(
    image_record: ImageRecord,
    styles,
    image_size: tuple[int, int],
    card_width: int,
) -> Table:
    """Build the left-side ImageRecord card."""
    main_img = pil_to_rl_image(image_record.image, max_size=image_size)

    metadata = Paragraph(
        f"""
        <b>Category:</b> {image_record.category}<br/>
        <b>Rotation:</b> {image_record.rotation}<br/>
        <b>Distance:</b> {image_record.distance}<br/>
        <b>Lighting:</b> {image_record.lighting}
        """,
        styles["Small"],
    )

    card = Table(
        [[main_img], [metadata]],
        colWidths=[card_width],
    )

    card.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 1, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (0, 0), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )

    return card


def _build_match_card(
    match_record: MatchRecord,
    match_index: int,
    styles,
    image_size: tuple[int, int],
    card_width: int,
) -> Table:
    """Build one color-coded MatchRecord card."""
    if match_record.matched_image is not None:
        match_img = pil_to_rl_image(match_record.matched_image, max_size=image_size)
    else:
        match_img = Paragraph("[No image]", styles["Small"])

    text = Paragraph(
        f"""
        <b>MatchRecord[{match_index}]</b><br/>
        <b>Category:</b> {match_record.matched_category}<br/>
        <b>Distance:</b> {_distance_text(match_record.nn_distance)}<br/>
        <b>Correct:</b> {match_record.correct}
        """,
        styles["Small"],
    )

    bg_color = _match_background(match_record.correct)

    card = Table(
        [[match_img], [text]],
        colWidths=[card_width],
    )

    card.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), bg_color),
                ("BOX", (0, 0), (-1, -1), 1, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (0, 0), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )

    return card


def _chunked(items: Sequence, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _build_matches_table(
    image_record: ImageRecord,
    styles,
    matches_per_row: int,
    match_card_width: int,
    match_image_size: tuple[int, int],
) -> Table:
    """Build the right-side MatchRecord area, wrapping cards into rows."""
    match_rows = []

    if not image_record.match_records:
        no_matches = Paragraph("No match records", styles["Small"])
        return Table([[no_matches]], colWidths=[match_card_width])

    for match_row_start in range(0, len(image_record.match_records), matches_per_row):
        match_row = image_record.match_records[
            match_row_start : match_row_start + matches_per_row
        ]

        row_cells = []
        for offset, match_record in enumerate(match_row):
            match_index = match_row_start + offset
            row_cells.append(
                _build_match_card(
                    match_record=match_record,
                    match_index=match_index,
                    styles=styles,
                    image_size=match_image_size,
                    card_width=match_card_width,
                )
            )

        while len(row_cells) < matches_per_row:
            row_cells.append("")

        match_rows.append(row_cells)

    matches_table = Table(
        match_rows,
        colWidths=[match_card_width] * matches_per_row,
        hAlign="LEFT",
    )

    matches_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    return matches_table

def build_summary_page(stats, config, styles):
    elements = []

    title_style = styles["Title"]

    body_style = ParagraphStyle(
        "Body",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=9,
        leading=12,
        alignment=TA_LEFT,
    )

    elements.append(Paragraph("Match Statistics Summary", title_style))
    elements.append(Spacer(1, 12))

    # Stats block
    stats_text = f"""
    Total Matches: {stats.total_matches}<br/>
    Total Correct: {stats.total_correct}<br/>
    Total Incorrect: {stats.total_incorrect}<br/>
    Percent Correct: {stats.percent_correct}<br/><br/>

    Highest Correct Distance: {stats.highest_correct}<br/>
    Lowest Correct Distance: {stats.lowest_correct}<br/>
    Average Correct Distance: {stats.average_correct}<br/><br/>

    Highest Incorrect Distance: {stats.highest_incorrect}<br/>
    Lowest Incorrect Distance: {stats.lowest_incorrect}<br/>
    Average Incorrect Distance: {stats.average_incorrect}
    """

    elements.append(Paragraph(stats_text, body_style))
    elements.append(Spacer(1, 20))

    # Config block (pretty printed)
    import pprint
    config_str = pprint.pformat(config, indent=2, width=80)

    config_text = f"<b>Configuration:</b><br/><br/><font name='Courier'>{config_str}</font>"

    elements.append(Paragraph(config_text, body_style))

    elements.append(PageBreak())

    return elements

def _build_image_record_row(
    image_record: ImageRecord,
    styles,
    main_card_width: int,
    matches_area_width: int,
    main_image_size: tuple[int, int],
    match_image_size: tuple[int, int],
    matches_per_row: int,
    match_card_width: int,
) -> Table:
    left_card = _build_main_record_card(
        image_record=image_record,
        styles=styles,
        image_size=main_image_size,
        card_width=main_card_width,
    )

    matches_table = _build_matches_table(
        image_record=image_record,
        styles=styles,
        matches_per_row=matches_per_row,
        match_card_width=match_card_width,
        match_image_size=match_image_size,
    )

    row_table = Table(
        [[left_card, matches_table]],
        colWidths=[main_card_width + 8, matches_area_width],
        hAlign="LEFT",
    )

    row_table.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 1, colors.black),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )

    return row_table


def create_image_record_match_pdf(
    image_records: Sequence[ImageRecord],
    timestamp: str,
    stats=None,
    config=None,
    records_per_page: int = 8,
    page_size=landscape(letter),
    main_image_size: tuple[int, int] = (95, 95),
    match_image_size: tuple[int, int] = (85, 85),
    main_card_width: int = 105,
    match_card_width: int = 100,
    matches_per_row: int = 5,
) -> str:
    """
    Create a paginated PDF visualization of ImageRecord objects and MatchRecord cards.

    Correct matches are light green.
    Incorrect matches are light red.
    Unknown/None correctness is gray.

    Args:
        image_records:
            Sequence of ImageRecord objects to visualize.
        output_path:
            PDF file path to create.
        records_per_page:
            Number of ImageRecord rows per PDF page.
            Lower this if rows are too tall for your match count.
        page_size:
            ReportLab page size. Defaults to landscape letter.
        main_image_size:
            Max thumbnail size for ImageRecord images.
        match_image_size:
            Max thumbnail size for MatchRecord images.
        main_card_width:
            Width of the left-side ImageRecord card.
        match_card_width:
            Width of each MatchRecord card.
        matches_per_row:
            Number of MatchRecord cards before wrapping to a new row.
    """
    project_root = Path(__file__).resolve().parents[2]
    
    # Build results directory path
    results_dir = project_root / "results" / timestamp
    results_dir.mkdir(exist_ok=True)

    output_path = results_dir / "match_results.pdf"

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=page_size,
        rightMargin=24,
        leftMargin=24,
        topMargin=24,
        bottomMargin=24,
    )

    base_styles = getSampleStyleSheet()
    styles = {
        "Normal": base_styles["Normal"],
        "Small": ParagraphStyle(
            "Small",
            parent=base_styles["Normal"],
            fontName="Helvetica",
            fontSize=6,
            leading=7,
            wordWrap="CJK",
        ),
        "Title": ParagraphStyle(
            "Title",
            parent=base_styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=16,
        ),
    }

    page_width, _ = page_size
    usable_width = page_width - doc.leftMargin - doc.rightMargin
    matches_area_width = usable_width - main_card_width - 20

    # Auto-adjust match card width if needed to prevent horizontal overflow.
    max_match_width = int(matches_area_width / max(matches_per_row, 1)) - 6
    effective_match_card_width = min(match_card_width, max_match_width)

    elements = []
    elements.extend(build_summary_page(stats, config, styles))

    total_records = len(image_records)
    for page_start in range(0, total_records, records_per_page):
        page_records = image_records[page_start : page_start + records_per_page]
        page_number = (page_start // records_per_page) + 1
        total_pages = ceil(total_records / records_per_page) if total_records else 1

        elements.append(
            Paragraph(
                f"Image Record Match Visualization — Page {page_number} of {total_pages}",
                styles["Title"],
            )
        )
        elements.append(Spacer(1, 8))

        for image_record in page_records:
            elements.append(
                _build_image_record_row(
                    image_record=image_record,
                    styles=styles,
                    main_card_width=main_card_width,
                    matches_area_width=matches_area_width,
                    main_image_size=main_image_size,
                    match_image_size=match_image_size,
                    matches_per_row=matches_per_row,
                    match_card_width=effective_match_card_width,
                )
            )
            elements.append(Spacer(1, 10))

        if page_start + records_per_page < total_records:
            elements.append(PageBreak())

    doc.build(elements)
    return results_dir
