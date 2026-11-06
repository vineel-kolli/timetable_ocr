
import json
import cv2

from paddleocr import PaddleOCR

from app.services.grid_detector import detect_timetable_grid
from app.services.cell_reconstructor import reconstruct_cells


IMAGE_PATH = "tests/fixtures/timetable.jpeg"

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(IMAGE_PATH)

# Detect timetable geometry.
grid = detect_timetable_grid(image)

# Use the saved benchmark OCR results.
# This isolates cell reconstruction from OCR inference.
with open("tests/fixtures/timetable_res.json") as file:
    ocr_data = json.load(file)

result = reconstruct_cells(grid, ocr_data)

print("\n========== CELL RECONSTRUCTION ==========")

for row in result["rows"]:
    print(f"\nRow {row['row_index']} ({row['y1']}–{row['y2']})")

    for cell in row["cells"]:
        print(
            f"P{cell['start_period']}-P{cell['end_period']} | "
            f"{cell['status']:<10} | "
            f"{cell['text'] or '[EMPTY]'} | "
            f"Confidence: {cell['confidence']}"
        )

with open("tests/cell_reconstruction_output.json", "w") as file:
    json.dump(result, file, indent=4)

print("\nSaved: tests/cell_reconstruction_output.json")
