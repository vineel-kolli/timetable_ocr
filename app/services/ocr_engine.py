
from threading import Lock
from typing import Any

import numpy as np
from paddleocr import PaddleOCR


class OCREngineError(Exception):
    """Raised when OCR initialization or inference fails."""


class OCREngine:
    """Manages one reusable PaddleOCR instance."""

    def __init__(self) -> None:
        self._ocr: PaddleOCR | None = None
        self._lock = Lock()

    def initialize(self) -> None:
        """Initialize PaddleOCR once."""

        if self._ocr is not None:
            return

        with self._lock:
            if self._ocr is not None:
                return

            try:
                self._ocr = PaddleOCR(
                    lang="en",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
            except Exception as exc:
                raise OCREngineError(
                    f"Could not initialize PaddleOCR: {exc}"
                ) from exc

    def predict(self, image: np.ndarray) -> dict[str, Any]:
        """Run OCR on an OpenCV image and return normalized results."""

        if self._ocr is None:
            raise OCREngineError(
                "OCR engine has not been initialized."
            )

        if image is None or image.size == 0:
            raise OCREngineError("Input image is empty.")

        with self._lock:
            try:
                results = self._ocr.predict(image)
            except Exception as exc:
                raise OCREngineError(
                    f"OCR inference failed: {exc}"
                ) from exc

        if not results:
            raise OCREngineError("OCR returned no results.")

        result = results[0]

        data = (
            result.json
            if hasattr(result, "json")
            else result
        )

        if isinstance(data, dict) and "res" in data:
            data = data["res"]

        required = ("rec_texts", "rec_scores", "rec_boxes")

        if not isinstance(data, dict) or not all(
            key in data for key in required
        ):
            raise OCREngineError(
                "OCR result does not contain the expected fields."
            )

        return {
            "rec_texts": data["rec_texts"],
            "rec_scores": data["rec_scores"],
            "rec_boxes": data["rec_boxes"],
        }

