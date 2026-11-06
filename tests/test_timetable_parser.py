
import json
import cv2

from app.services.grid_detector import detect_timetable_grid
from app.services.cell_reconstructor import reconstruct_cells
from app.services.timetable_parser import parse_timetable


image = cv2.imread("tests/fixtures/timetable.jpeg")

if image is None:
    raise FileNotFoundError("Benchmark image not found.")

grid = detect_timetable_grid(image)

with open("tests/fixtures/timetable_res.json") as file:
    ocr_data = json.load(file)

reconstructed = reconstruct_cells(grid, ocr_data)
parsed = parse_timetable(reconstructed)

recognized = 0
empty = 0

for row in parsed["rows"]:
    print(f"\nRow {row['row_index']}")

    for entry in row["entries"]:
        if entry["status"] == "recognized":
            recognized += 1
        else:
            empty += 1

        print(
            f"P{entry['start_period']}-P{entry['end_period']} | "
            f"{entry['subject'] or '[EMPTY]'} | "
            f"Faculty: {entry['faculty']} | "
            f"Room: {entry['classroom']}"
        )

print("\n========== SUMMARY ==========")
print("Recognized:", recognized)
print("Empty:", empty)

with open("tests/parsed_output.json", "w") as file:
    json.dump(parsed, file, indent=4)

print("\nSaved: tests/parsed_output.json")
