from dataclasses import dataclass
from typing import Any

from app.services.grid_detector import GridDetection


@dataclass(frozen=True)
class CellReconstructionConfig:
    """
    Configuration for mapping detected grid cells to periods.

    Period positions are not hard-coded in pixels. They are read from
    the period numbers printed in the timetable header row, so the
    same logic works for different image sizes and layouts.
    """

    valid_periods: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9)


class CellReconstructionError(Exception):
    """Raised when OCR results cannot be mapped to the detected grid."""


def _validate_ocr_data(ocr_data: dict[str, Any]) -> None:
    required_fields = ("rec_texts", "rec_scores", "rec_boxes")

    missing = [
        field for field in required_fields
        if field not in ocr_data
    ]

    if missing:
        raise CellReconstructionError(
            f"Missing OCR fields: {', '.join(missing)}"
        )

    lengths = [
        len(ocr_data[field])
        for field in required_fields
    ]

    if len(set(lengths)) != 1:
        raise CellReconstructionError(
            "OCR texts, scores and boxes have different lengths."
        )


def _detect_period_centers(
    ocr_data: dict[str, Any],
    header_bottom: int,
    config: CellReconstructionConfig,
) -> dict[int, float]:
    """Find the x-center of each period number printed in the header row."""

    centers: dict[int, tuple[float, float]] = {}  # period -> (x, score)
    valid = set(config.valid_periods)

    for text, score, box in zip(
        ocr_data["rec_texts"],
        ocr_data["rec_scores"],
        ocr_data["rec_boxes"],
    ):
        text = str(text).strip()

        if not text.isdigit() or int(text) not in valid:
            continue

        if len(box) != 4:
            continue

        x1, y1, x2, y2 = map(float, box)

        # Digits below the header belong to body rows, not the header.
        if (y1 + y2) / 2 >= header_bottom:
            continue

        period = int(text)

        # Keep the highest-confidence reading if a digit appears twice.
        if period not in centers or score > centers[period][1]:
            centers[period] = ((x1 + x2) / 2, float(score))

    result = {period: x for period, (x, _) in centers.items()}

    missing = sorted(valid - set(result))
    if missing:
        raise CellReconstructionError(
            "Could not read all period numbers from the header row. "
            f"Missing: {missing}"
        )

    # Period numbers must increase from left to right.
    ordered = [result[period] for period in sorted(result)]
    if ordered != sorted(ordered):
        raise CellReconstructionError(
            "Period numbers in the header row are not in left-to-right order."
        )

    return result


def _period_span(
    left: int,
    right: int,
    period_centers: dict[int, float],
) -> tuple[int, int] | None:
    """
    Find the periods covered by a grid cell.

    A period belongs to a cell only if the center of its header digit
    lies inside the cell. Each digit therefore belongs to exactly one
    cell, so spans cannot overlap. Cells with no digit (the day column,
    the lunch column) are skipped.
    """

    occupied = [
        period
        for period, center in period_centers.items()
        if left <= center < right
    ]

    if not occupied:
        return None

    return min(occupied), max(occupied)


def _find_text_regions(
    left: int,
    right: int,
    top: int,
    bottom: int,
    ocr_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Find OCR regions whose centers lie inside a grid cell."""

    matches = []

    for text, score, box in zip(
        ocr_data["rec_texts"],
        ocr_data["rec_scores"],
        ocr_data["rec_boxes"],
    ):
        if not str(text).strip():
            continue

        if len(box) != 4:
            continue

        x1, y1, x2, y2 = map(float, box)

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # Use half-open intervals to avoid assigning a region
        # to two neighboring cells at a shared boundary.
        if not (left <= center_x < right):
            continue

        if not (top <= center_y < bottom):
            continue

        matches.append({
            "text": str(text).strip(),
            "confidence": round(float(score), 4),
            "box": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            },
        })

    return sorted(
        matches,
        key=lambda item: item["box"]["x1"],
    )


def reconstruct_cells(
    grid: GridDetection,
    ocr_data: dict[str, Any],
    config: CellReconstructionConfig = CellReconstructionConfig(),
) -> dict[str, Any]:
    """
    Reconstruct timetable cells from grid geometry and OCR regions.

    Each detected cell is mapped to one or more periods using the
    positions of the period numbers in the header row.
    Empty cells are retained in the output.
    """

    _validate_ocr_data(ocr_data)

    if len(grid.horizontal_lines) < 2:
        raise CellReconstructionError("Grid has no header row.")

    header_bottom = grid.horizontal_lines[1]
    period_centers = _detect_period_centers(
        ocr_data,
        header_bottom,
        config,
    )

    rows = []

    for row in grid.row_boundaries:
        boundaries = row["aligned_boundaries"]
        cells = []

        for left, right in zip(boundaries, boundaries[1:]):
            span = _period_span(left, right, period_centers)

            # Day column, lunch column and thin slivers have no period.
            if span is None:
                continue

            start_period, end_period = span

            text_regions = _find_text_regions(
                left,
                right,
                row["y1"],
                row["y2"],
                ocr_data,
            )

            recognized_text = " ".join(
                region["text"] for region in text_regions
            )

            confidence = (
                min(region["confidence"] for region in text_regions)
                if text_regions
                else None
            )

            cells.append({
                "start_period": start_period,
                "end_period": end_period,
                "text": recognized_text or None,
                "confidence": confidence,
                "status": (
                    "recognized" if text_regions else "empty"
                ),
                "text_regions": text_regions,
                "cell_bounds": {
                    "x1": left,
                    "x2": right,
                    "y1": row["y1"],
                    "y2": row["y2"],
                },
            })

        cells.sort(key=lambda cell: cell["start_period"])

        # Safety check: periods within a row must never overlap.
        for a, b in zip(cells, cells[1:]):
            if a["end_period"] >= b["start_period"]:
                raise CellReconstructionError(
                    f"Row {row['row_index']}: overlapping periods "
                    f"{a['start_period']}-{a['end_period']} and "
                    f"{b['start_period']}-{b['end_period']}."
                )

        rows.append({
            "row_index": row["row_index"],
            "y1": row["y1"],
            "y2": row["y2"],
            "cells": cells,
        })

    return {
        "image_width": grid.image_width,
        "image_height": grid.image_height,
        "rows": rows,
    }