"""Combine the two standalone assignment reports for the form's PDF upload."""

import tempfile
from pathlib import Path

from pypdf import PdfReader, PdfWriter


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    reports = [
        ("Assignment 2a - Forced alignment", ROOT / "assignment-2a/report.pdf"),
        ("Assignment 2b - Multimodal emotion recognition", ROOT / "assignment-2b/report.pdf"),
    ]
    writer = PdfWriter()
    for title, source in reports:
        reader = PdfReader(source)
        if not reader.pages:
            raise ValueError(f"Report is empty: {source}")
        start = len(writer.pages)
        writer.append(reader, import_outline=False)
        writer.add_outline_item(title=title, page_number=start)
    writer.add_metadata({
        "/Title": "IIIT Hyderabad Assignment 2 - Parts 2a and 2b",
        "/Author": "Parichay Mago",
    })
    with tempfile.TemporaryDirectory(dir=ROOT) as directory:
        temporary = Path(directory) / "report.pdf"
        writer.write(temporary)
        if len(PdfReader(temporary).pages) != len(writer.pages):
            raise ValueError("Combined report page count differs from the source reports")
        temporary.replace(ROOT / "report.pdf")
    print(f"Wrote {len(writer.pages)} pages to {ROOT / 'report.pdf'}")


if __name__ == "__main__":
    main()
