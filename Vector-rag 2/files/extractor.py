"""
extractor.py — يستخرج النص من TXT أو PDF
"""

from pathlib import Path


def extract_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return _read_txt(path)
    elif suffix == ".pdf":
        return _read_pdf(path)
    else:
        raise ValueError(f"نوع الملف مش مدعوم: {suffix} (بس .txt و .pdf)")


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(path))
        pages = [page.get_text() for page in doc]
        doc.close()
        return "\n".join(pages)
    except ImportError:
        raise ImportError(
            "محتاج تثبّت PyMuPDF:\n  pip install pymupdf"
        )
