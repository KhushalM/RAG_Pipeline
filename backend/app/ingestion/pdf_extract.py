import pymupdf
from typing import List


def extract_text_from_pdf(file_path: str) -> str:
    """Extract plain text from a single PDF file using PyMuPDF.

    Returns a single string containing the concatenated text of all pages.
    """
    doc = pymupdf.open(file_path)
    pages_text: List[str] = []
    for page in doc:
        text = page.get_text()
        if text:
            pages_text.append(text.strip())
    doc.close()
    return "\n".join(pages_text)



