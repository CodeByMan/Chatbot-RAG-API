from pypdf import PdfReader
from typing import BinaryIO


def extract_text_from_pdf(pdf_file: BinaryIO, max_pages: int = 50) -> str:
    """Extract text from PDF file"""
    pdf_reader = PdfReader(pdf_file)
    if len(pdf_reader.pages) > max_pages:
        raise ValueError(f"PDF files must contain {max_pages} pages or fewer")

    text = ""

    for page in pdf_reader.pages:
        text += (page.extract_text() or "") + "\n"

    return text.strip()