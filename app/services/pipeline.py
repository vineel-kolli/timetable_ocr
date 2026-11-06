
from typing import Any

import numpy as np

from app.services.ocr_engine import OCREngine
from app.services.grid_detector import detect_timetable_grid
from app.services.cell_reconstructor import reconstruct_cells
from app.services.timetable_parser import parse_timetable


class TimetablePipelineError(Exception):
    """Raised when timetable processing fails."""


class TimetableOCRPipeline:
    """Coordinates OCR, grid detection, cell reconstruction and parsing."""

    def __init__(self, ocr_engine: OCREngine) -> None:
        self.ocr_engine = ocr_engine

    def process(self, image: np.ndarray) -> dict[str, Any]:
        """
        Process a timetable image and return structured timetable data.
        """

        if image is None or not isinstance(image, np.ndarray):
            raise TimetablePipelineError("A valid image is required.")

        if image.size == 0:
            raise TimetablePipelineError("The input image is empty.")

        try:
            # Stage 1: Extract text, confidence and bounding boxes.
            ocr_data = self.ocr_engine.predict(image)

            # Stage 2: Detect timetable rows and column boundaries.
            grid = detect_timetable_grid(image)

            # Stage 3: Map OCR regions to timetable cells.
            reconstructed = reconstruct_cells(grid, ocr_data)

            # Stage 4: Extract subjects, faculty and classrooms.
            parsed = parse_timetable(reconstructed)

            return {
                "success": True,
                "source": parsed["source"],
                "image_width": parsed["image_width"],
                "image_height": parsed["image_height"],
                "rows": parsed["rows"],
                "diagnostics": {
                    "ocr_regions": len(ocr_data["rec_texts"]),
                    "detected_rows": len(grid.row_boundaries),
                    "header_boundaries": len(grid.header_boundaries),
                    "recognized_cells": sum(
                        entry["status"] == "recognized"
                        for row in parsed["rows"]
                        for entry in row["entries"]
                    ),
                    "empty_cells": sum(
                        entry["status"] == "empty"
                        for row in parsed["rows"]
                        for entry in row["entries"]
                    ),
                },
            }

        except Exception as exc:
            raise TimetablePipelineError(
                f"Timetable processing failed: {exc}"
            ) from exc
