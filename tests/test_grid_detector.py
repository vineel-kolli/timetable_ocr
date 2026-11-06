from pathlib import Path

import cv2

from app.services.grid_detector import detect_timetable_grid


def test_detect_timetable_grid():
    """Verify that the timetable grid detector finds valid boundaries."""

    image_path = (
        Path(__file__).parent / "fixtures" / "timetable.jpeg"
    )

    image = cv2.imread(str(image_path))

    assert image is not None, "Test timetable image could not be loaded"

    result = detect_timetable_grid(image)

    # Basic image validation
    assert result.image_width == image.shape[1]
    assert result.image_height == image.shape[0]

    # Horizontal boundaries must be detected
    assert len(result.horizontal_lines) >= 3

    # Header column boundaries must be detected
    assert len(result.header_boundaries) >= 3

    # Row boundaries must be present
    assert len(result.row_boundaries) >= 1

    # Every detected row must have valid coordinates
    for row in result.row_boundaries:
        assert row["y1"] < row["y2"]

    print("\nGrid detection results:")
    print("Horizontal lines:", result.horizontal_lines)
    print("Header boundaries:", result.header_boundaries)
    print("Detected rows:", len(result.row_boundaries))
    print("Diagnostics:", result.diagnostics)