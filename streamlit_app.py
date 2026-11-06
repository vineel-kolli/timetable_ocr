import cv2
import numpy as np
import streamlit as st

from app.services.ocr_engine import OCREngine
from app.services.pipeline import (
    TimetableOCRPipeline,
    TimetablePipelineError,
)


MAX_IMAGE_PIXELS = 10_000_000
MAX_IMAGE_DIMENSION = 4000
MAX_FILE_SIZE = 5 * 1024 * 1024


st.set_page_config(
    page_title="Attendly Timetable OCR",
    page_icon="📚",
    layout="wide",
)


@st.cache_resource
def load_pipeline():
    """Initialize PaddleOCR once and reuse it across app reruns."""
    print("Initializing Attendly OCR service...")

    ocr_engine = OCREngine()
    ocr_engine.initialize()

    pipeline = TimetableOCRPipeline(ocr_engine)

    print("Attendly OCR service is ready.")
    return pipeline


st.title("📚 Attendly Timetable OCR")

st.write(
    "Upload a timetable image to extract subjects, "
    "faculty, classrooms and period information."
)

uploaded_file = st.file_uploader(
    "Upload timetable image",
    type=["png", "jpg", "jpeg"],
)

if uploaded_file is not None:

    if len(uploaded_file.getvalue()) > MAX_FILE_SIZE:
        st.error("Image file exceeds the 5 MB limit.")
        st.stop()

    image_bytes = np.frombuffer(
        uploaded_file.getvalue(),
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        image_bytes,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        st.error("Invalid image. Please upload a valid image.")
        st.stop()

    height, width = image.shape[:2]

    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        st.error("Image dimensions exceed the allowed limit.")
        st.stop()

    if width * height > MAX_IMAGE_PIXELS:
        st.error("Image contains too many pixels.")
        st.stop()

    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        caption="Uploaded timetable",
        use_container_width=True,
    )

    if st.button("Extract Timetable", type="primary"):

        try:
            with st.spinner("Processing timetable with PaddleOCR..."):

                pipeline = load_pipeline()
                result = pipeline.process(image)

            st.success("Timetable processed successfully!")

            st.subheader("Extracted Timetable")
            st.json(result)

        except TimetablePipelineError as exc:
            print(f"Timetable processing error: {exc}")
            st.error("Could not process this timetable image.")

        except Exception as exc:
            print(f"Unexpected OCR error: {exc}")
            st.error("An unexpected error occurred during OCR.")