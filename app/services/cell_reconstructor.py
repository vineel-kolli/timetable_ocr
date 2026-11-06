
from dataclasses import dataclass
from typing import Any

from app.services.grid_detector import GridDetection


@dataclass(frozen=True)
class PeriodColumn:
    """Maps an image column interval to a timetable period."""

    period: int
    left: int
    right: int


@dataclass(frozen=True)
class CellReconstructionConfig:
    """
    Configuration for mapping detected grid cells to periods.

    These column mappings describe the current benchmark timetable.
    They are configurable because different timetable layouts may
    have different numbers of periods or lunch positions.
    """

    period_columns: tuple[PeriodColumn, ...] = (
        PeriodColumn(1, 71, 125),
        PeriodColumn(2, 125, 179),
        PeriodColumn(3, 179, 255),
        PeriodColumn(4, 255, 310),
        PeriodColumn(5, 343, 402),
        PeriodColumn(6, 402, 456),
        PeriodColumn(7, 456, 511),
        PeriodColumn(8, 511, 568),
        PeriodColumn(9, 568, 636),
    )

    day_column_right: int = 71
    lunch_region: tuple[int, int] = (310, 343)


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


def _period_span(
    left: int,
    right: int,
    period_columns: tuple[PeriodColumn, ...],
) -> tuple[int, int] | None:
    """Find all timetable periods overlapped by a grid cell."""

    occupied = []

    for column in period_columns:
        overlap = min(right, column.right) - max(left, column.left)

        if overlap > 0:
            occupied.append(column.period)

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

    Each detected cell is mapped to one or more periods.
    Empty cells are retained in the output.
    """

    _validate_ocr_data(ocr_data)

    if not grid.header_boundaries:
        raise CellReconstructionError(
            "Grid has no reference header boundaries."
        )

    rows = []

    for row in grid.row_boundaries:
        boundaries = row["aligned_boundaries"]

        cells = []

        for left, right in zip(boundaries, boundaries[1:]):

            # Ignore the day-name column.
            if right <= config.day_column_right:
                continue

            # Ignore the lunch column.
            lunch_left, lunch_right = config.lunch_region

            if left >= lunch_left and right <= lunch_right:
                continue

            span = _period_span(
                left,
                right,
                config.period_columns,
            )

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
