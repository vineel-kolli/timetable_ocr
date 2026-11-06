
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


class GridDetectionError(Exception):
    """Raised when a reliable timetable grid cannot be detected."""


@dataclass(frozen=True)
class GridConfig:
    horizontal_kernel_ratio: float = 1 / 30
    vertical_kernel_ratio: float = 1 / 10
    line_projection_threshold: float = 0.20
    row_projection_threshold: float = 0.35
    alignment_tolerance_ratio: float = 0.0125
    minimum_rows: int = 2
    minimum_columns: int = 2


@dataclass
class GridDetection:
    image_width: int
    image_height: int
    horizontal_lines: list[int]
    header_boundaries: list[int]
    row_boundaries: list[dict]
    diagnostics: dict


def _group_active_positions(active: np.ndarray) -> list[int]:
    """Convert consecutive active projection positions into line centers."""

    positions = []
    start: Optional[int] = None

    for index, value in enumerate(active):
        if value and start is None:
            start = index

        elif not value and start is not None:
            positions.append((start + index - 1) // 2)
            start = None

    if start is not None:
        positions.append((start + len(active) - 1) // 2)

    return positions


def _projection_lines(
    image: np.ndarray,
    axis: int,
    threshold_ratio: float,
) -> list[int]:
    """Detect line positions using a binary image projection."""

    projection = np.sum(image > 0, axis=axis)

    if axis == 0:
        reference_size = image.shape[0]  # Height
    elif axis == 1:
        reference_size = image.shape[1]  # Width
    else:
        raise ValueError("Axis must be 0 or 1.")

    active = projection > reference_size * threshold_ratio

    return _group_active_positions(active)

def _extract_lines(
    binary: np.ndarray,
    horizontal_ratio: float,
    vertical_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:

    height, width = binary.shape

    horizontal_length = max(10, round(width * horizontal_ratio))
    vertical_length = max(10, round(height * vertical_ratio))

    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (horizontal_length, 1),
    )

    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, vertical_length),
    )

    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        horizontal_kernel,
    )

    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        vertical_kernel,
    )

    return horizontal, vertical


def _detect_row_vertical_boundaries(
    vertical_image: np.ndarray,
    y1: int,
    y2: int,
    threshold_ratio: float,
) -> list[int]:

    row_image = vertical_image[y1:y2, :]

    projection = np.sum(row_image > 0, axis=0)

    active = projection > (y2 - y1) * threshold_ratio

    return _group_active_positions(active)


def _align_boundaries(
    detected: list[int],
    reference: list[int],
    tolerance: int,
) -> list[int]:

    aligned = []

    for position in detected:
        nearest = min(
            reference,
            key=lambda boundary: abs(boundary - position),
        )

        if abs(nearest - position) <= tolerance:
            aligned.append(nearest)

    return sorted(set(aligned))

def _merge_boundary_candidates(
    candidates: list[int],
    tolerance: int,
) -> list[int]:
    """Merge nearby vertical-line positions into global boundaries."""

    if not candidates:
        return []

    positions = sorted(candidates)
    groups = [[positions[0]]]

    for position in positions[1:]:
        if position - groups[-1][-1] <= tolerance:
            groups[-1].append(position)
        else:
            groups.append([position])

    return [
        round(sum(group) / len(group))
        for group in groups
    ]

def detect_timetable_grid(
    image: np.ndarray,
    config: GridConfig = GridConfig(),
) -> GridDetection:
    """
    Detect timetable rows and reference column boundaries.

    The function accepts an OpenCV BGR or grayscale image.
    It does not perform OCR or timetable text reconstruction.
    """

    if image is None or not isinstance(image, np.ndarray):
        raise GridDetectionError("A valid image is required.")

    if image.size == 0:
        raise GridDetectionError("The supplied image is empty.")

    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise GridDetectionError("Unsupported image format.")

    height, width = gray.shape

    if height < 30 or width < 100:
        raise GridDetectionError("Image is too small for grid detection.")

    binary = cv2.adaptiveThreshold(
        cv2.bitwise_not(gray),
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        15,
        -2,
    )

    horizontal, vertical = _extract_lines(
        binary,
        config.horizontal_kernel_ratio,
        config.vertical_kernel_ratio,
    )

    horizontal_lines = _projection_lines(
        horizontal,
        axis=1,
        threshold_ratio=config.line_projection_threshold,
    )
    # Ignore a detected line at the final image edge.
    # It may be caused by the image crop boundary rather than
    # an actual timetable row boundary.
    if horizontal_lines and horizontal_lines[-1] >= height - 2:
        horizontal_lines.pop()
    # Include the image bottom as a boundary when the final
    # timetable row extends to the bottom of the cropped image.
    if horizontal_lines and height - 1 - horizontal_lines[-1] > 3:
        horizontal_lines.append(height - 1)
    if len(horizontal_lines) < config.minimum_rows + 1:
        raise GridDetectionError(
            "Not enough horizontal lines were detected."
        )

    # The first horizontal band is treated as the table header.
    header_top = horizontal_lines[0]
    header_bottom = horizontal_lines[1]

    header_image = vertical[header_top:header_bottom, :]

    header_positions = _projection_lines(
        header_image,
        axis=0,
        threshold_ratio=config.line_projection_threshold,
    )

    if len(header_positions) < config.minimum_columns + 1:
        raise GridDetectionError(
            "Not enough reference column boundaries were detected."
        )

    tolerance = max(
        2,
        round(width * config.alignment_tolerance_ratio),
    )

    row_boundaries = []

    for row_index, (y1, y2) in enumerate(
        zip(horizontal_lines[1:-1], horizontal_lines[2:])
    ):
        detected = _detect_row_vertical_boundaries(
            vertical,
            y1,
            y2,
            config.row_projection_threshold,
        )

        aligned = _align_boundaries(
            detected,
            header_positions,
            tolerance,
        )

        row_boundaries.append({
            "row_index": row_index,
            "y1": y1,
            "y2": y2,
            "detected_boundaries": detected,
            "aligned_boundaries": aligned,
        })

    return GridDetection(
        image_width=width,
        image_height=height,
        horizontal_lines=horizontal_lines,
        header_boundaries=header_positions,
        row_boundaries=row_boundaries,
        diagnostics={
            "horizontal_line_count": len(horizontal_lines),
            "header_boundary_count": len(header_positions),
            "alignment_tolerance": tolerance,
        },
    )
