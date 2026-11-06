
import re
from typing import Any


class TimetableParsingError(Exception):
    """Raised when reconstructed timetable data is invalid."""


# Examples: U-407, U-405, LAB-1
ROOM_PATTERN = re.compile(
    r"\b(?:U-\d+|LAB-\d+)\b",
    re.IGNORECASE,
)

# Examples: PSV, PMA, KVS, NN, PB
FACULTY_CODE_PATTERN = re.compile(
    r"^[A-Z]{2,5}$",
    re.IGNORECASE,
)


def _extract_classroom(text: str) -> tuple[str | None, str]:
    """Extract a classroom identifier and remove it from the text."""

    match = ROOM_PATTERN.search(text)

    if not match:
        return None, text

    classroom = match.group(0).upper()
    remaining = (
        text[:match.start()] + " " + text[match.end():]
    )

    return classroom, remaining.strip()


def _extract_faculty(
    text: str,
) -> tuple[list[str], str]:
    """
    Extract parenthesized faculty codes.

    Only remove a parenthesized group if every item looks like
    a faculty code. Otherwise preserve it as part of the subject.
    """

    faculty = []
    remaining = text

    matches = list(re.finditer(r"\(([^()]*)\)", text))

    # Process from right to left so string positions remain valid.
    for match in reversed(matches):
        values = [
            item.strip()
            for item in match.group(1).split(",")
            if item.strip()
        ]

        if not values:
            continue

        if not all(
            FACULTY_CODE_PATTERN.fullmatch(value)
            for value in values
        ):
            continue

        faculty = [value.upper() for value in values]

        remaining = (
            remaining[:match.start()]
            + " "
            + remaining[match.end():]
        )

    return faculty, remaining.strip()


def parse_cell(cell: dict[str, Any]) -> dict[str, Any]:
    """Convert one reconstructed cell into a structured entry."""

    if cell.get("status") == "empty" or not cell.get("text"):
        return {
            "start_period": cell["start_period"],
            "end_period": cell["end_period"],
            "subject": None,
            "faculty": [],
            "classroom": None,
            "original_text": None,
            "confidence": None,
            "status": "empty",
        }

    original = str(cell["text"]).strip()

    classroom, text_without_room = _extract_classroom(original)

    faculty, subject = _extract_faculty(text_without_room)

    subject = re.sub(r"\s+", " ", subject).strip(" ,")

    if not subject:
        subject = original

    return {
        "start_period": cell["start_period"],
        "end_period": cell["end_period"],
        "subject": subject,
        "faculty": faculty,
        "classroom": classroom,
        "original_text": original,
        "confidence": cell.get("confidence"),
        "status": "recognized",
    }


def parse_timetable(
    reconstructed: dict[str, Any],
) -> dict[str, Any]:
    """Parse all reconstructed timetable rows."""

    if "rows" not in reconstructed:
        raise TimetableParsingError(
            "Reconstructed data must contain rows."
        )

    parsed_rows = []

    for row in reconstructed["rows"]:
        if "cells" not in row:
            raise TimetableParsingError(
                f"Row {row.get('row_index')} has no cells."
            )

        entries = [
            parse_cell(cell)
            for cell in row["cells"]
        ]

        parsed_rows.append({
            "row_index": row["row_index"],
            "entries": entries,
        })

    return {
        "source": "PaddleOCR + OpenCV",
        "image_width": reconstructed["image_width"],
        "image_height": reconstructed["image_height"],
        "rows": parsed_rows,
    }
