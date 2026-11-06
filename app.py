import os

import cv2
import gradio as gr
import numpy as np

from app.services.ocr_engine import OCREngine
from app.services.pipeline import (
    TimetableOCRPipeline,
    TimetablePipelineError,
)


MAX_IMAGE_PIXELS = 10_000_000
MAX_IMAGE_DIMENSION = 4000


# Initialize the OCR engine once when the application starts.
print("Initializing Attendly OCR service...")

ocr_engine = OCREngine()
ocr_engine.initialize()

pipeline = TimetableOCRPipeline(ocr_engine)

print("Attendly OCR service is ready.")


def predict_timetable(image: np.ndarray):
    """Process an uploaded timetable image."""

    if image is None:
        raise gr.Error("Please upload a timetable image.")

    if not isinstance(image, np.ndarray) or image.size == 0:
        raise gr.Error("Invalid image.")

    height, width = image.shape[:2]

    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise gr.Error("Image dimensions exceed the allowed limit.")

    if width * height > MAX_IMAGE_PIXELS:
        raise gr.Error("Image contains too many pixels.")

    # Gradio supplies RGB images; OpenCV processing expects BGR.
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    try:
        result = pipeline.process(bgr_image)
        return result

    except TimetablePipelineError as exc:
        print(f"Timetable processing error: {exc}")
        raise gr.Error("Could not process this timetable image.")

    except Exception as exc:
        print(f"Unexpected OCR error: {exc}")
        raise gr.Error("An unexpected error occurred during OCR.")


with gr.Blocks(title="Attendly Timetable OCR") as demo:

    gr.Markdown(
        """
        # Attendly Timetable OCR

        Upload a timetable image to extract subjects, faculty,
        classrooms and period information.
        """
    )

    with gr.Row():
        image_input = gr.Image(
            label="Upload timetable",
            type="numpy",
            image_mode="RGB",
            sources=["upload"],
        )

        json_output = gr.JSON(
            label="Extracted timetable",
        )

    predict_button = gr.Button(
        "Extract Timetable",
        variant="primary",
    )

    predict_button.click(
        fn=predict_timetable,
        inputs=image_input,
        outputs=json_output,
        api_name="predict_timetable",
        concurrency_limit=1,
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        max_file_size="5mb",
    )
