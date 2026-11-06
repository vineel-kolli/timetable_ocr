from pathlib import Path

import cv2

from app.services.grid_detector import detect_timetable_grid

import numpy as np

from app.services.grid_detector import _extract_lines

IMAGE_PATH = Path("tests/fixtures/csitb.jpeg")
OUTPUT_PATH = Path("tests/fixtures/csitb_grid_debug.png")

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(IMAGE_PATH)

from app.services.grid_detector import GridConfig

result = detect_timetable_grid(
    image,
    GridConfig(vertical_kernel_ratio=1 / 4),
)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

binary = cv2.adaptiveThreshold(
    cv2.bitwise_not(gray),
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    15,
    -2,
)

horizontal_mask, vertical_mask = _extract_lines(
    binary,
    1 / 30,
    1 / 4,
)
# Measure how much vertical-line evidence exists
# at each horizontal position across the full image.
vertical_projection = np.sum(vertical_mask > 0, axis=0)
print("\nStrong row-wise vertical boundaries:")

for row_index, (y1, y2) in enumerate(
    zip(result.horizontal_lines[1:-1], result.horizontal_lines[2:])
):
    row_mask = vertical_mask[y1:y2, :]
    projection = np.sum(row_mask > 0, axis=0)

    # Require support across at least 65% of the row height.
    threshold = (y2 - y1) * 0.65
    active = projection > threshold

    positions = []

    start = None

    for x, value in enumerate(active):
        if value and start is None:
            start = x

        elif not value and start is not None:
            center = (start + x - 1) // 2
            positions.append(center)
            start = None

    if start is not None:
        positions.append((start + len(active) - 1) // 2)

    print(f"Row {row_index}: {positions}")
print("\nFull-height vertical projection:")

for x, support in enumerate(vertical_projection):
    if support >= 5:
        print(f"x={x}: support={support}")

cv2.imwrite(
    "tests/fixtures/csitb_vertical_mask.png",
    vertical_mask,
)

cv2.imwrite(
    "tests/fixtures/csitb_horizontal_mask.png",
    horizontal_mask,
)

debug = image.copy()

# Draw horizontal boundaries
for y in result.horizontal_lines:
    cv2.line(debug, (0, y), (result.image_width, y), (0, 0, 255), 1)

# Draw header reference boundaries
for x in result.header_boundaries:
    cv2.line(debug, (x, 0), (x, result.image_height), (0, 255, 0), 1)

# Draw row-specific detected and aligned boundaries
for row in result.row_boundaries:
    y1 = row["y1"]
    y2 = row["y2"]

    for x in row["detected_boundaries"]:
        cv2.line(debug, (x, y1), (x, y2), (255, 0, 0), 1)

    for x in row["aligned_boundaries"]:
        cv2.circle(debug, (x, (y1 + y2) // 2), 3, (0, 255, 255), -1)

cv2.imwrite(str(OUTPUT_PATH), debug)

print("Saved:", OUTPUT_PATH)
print("Horizontal lines:", result.horizontal_lines)
print("Header boundaries:", result.header_boundaries)

for row in result.row_boundaries:
    print(
        f"Row {row['row_index']}: "
        f"Y={row['y1']}–{row['y2']} | "
        f"Detected={row['detected_boundaries']} | "
        f"Aligned={row['aligned_boundaries']}"
    )