"""ATGuide.pdf -> ATGuide.txt (vektör/taranmış sayfalar için OCR)."""

import sys

import cv2
import fitz
import numpy as np
from rapidocr_onnxruntime import RapidOCR


def pixmap_to_bgr(pix: fitz.Pixmap) -> np.ndarray:
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def ocr_lines(ocr: RapidOCR, img: np.ndarray) -> list[str]:
    res, _ = ocr(img)
    if not res:
        return []
    def sort_key(item):
        box = item[0]
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        return (sum(ys) / len(ys), sum(xs) / len(xs))
    ordered = sorted(res, key=sort_key)
    return [item[1] for item in ordered]


def main() -> int:
    pdf_path = "ATGuide.pdf"
    out_path = "ATGuide.txt"
    zoom = 2.0
    mat = fitz.Matrix(zoom, zoom)

    doc = fitz.open(pdf_path)
    ocr = RapidOCR()
    chunks: list[str] = []

    for i in range(len(doc)):
        pix = doc[i].get_pixmap(matrix=mat, alpha=False)
        img = pixmap_to_bgr(pix)
        lines = ocr_lines(ocr, img)
        body = "\n".join(lines) if lines else ""
        chunks.append(f"\n\n--- sayfa {i + 1} ---\n\n{body}")
        print(f"sayfa {i + 1}/{len(doc)} satır: {len(lines)}", flush=True)

    doc.close()
    text = "".join(chunks).lstrip("\n")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Yazıldı: {out_path} ({len(text)} karakter)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
