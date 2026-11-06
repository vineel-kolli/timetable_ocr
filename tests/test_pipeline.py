
import json

import cv2

from app.services.ocr_engine import OCREngine
from app.services.pipeline import TimetableOCRPipeline


IMAGE_PATH = "tests/fixtures/timetable.jpeg"
EXPECTED_PATH = "tests/fixtures/expected_parsed_output.json"

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(IMAGE_PATH)

# Initialize OCR only once.
engine = OCREngine()
engine.initialize()

pipeline = TimetableOCRPipeline(engine)

# Run the complete pipeline using real PaddleOCR inference.
result = pipeline.process(image)

print("\n========== PIPELINE DIAGNOSTICS ==========")
print(json.dumps(result["diagnostics"], indent=4))

print("\n========== RECONSTRUCTED TIMETABLE ==========")

for row in result["rows"]:
    print(f"\nRow {row['row_index']}")

    for entry in row["entries"]:
        print(
            f"P{entry['start_period']}-P{entry['end_period']} | "
            f"{entry['subject'] or '[EMPTY]'} | "
            f"Faculty: {entry['faculty']} | "
            f"Room: {entry['classroom']}"
        )

# Save the actual result.
with open("tests/pipeline_output.json", "w") as file:
    json.dump(result, file, indent=4)

print("\nSaved: tests/pipeline_output.json")
