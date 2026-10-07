
#!/usr/bin/env python3
"""
Rajasthan Voter PDF -> CSV + Excel

Columns:
SR. NO | VOTER_ID | NAME | FATHER'S NAME | HOUSE NUMBER | AGE | GENDER

Designed for the Rajasthan municipal voter-list layout used in the supplied
PDF: 3 voter cards per row and 9 rows per data page.

The parser uses the PDF text layer for VOTER_ID / AGE / GENDER / HOUSE NUMBER
when those values are available, and Hindi OCR on the left side of each voter
card for NAME / PARENT NAME. The left-side crop excludes the "Photo is
Available" area, which substantially reduces OCR noise.

Install:
    pip install pymupdf opencv-python pytesseract pandas openpyxl

Ubuntu/Debian:
    sudo apt-get install tesseract-ocr tesseract-ocr-hin

Run:
    python voter_pdf_parser.py input.pdf
    python voter_pdf_parser.py input.pdf -o voters
    python voter_pdf_parser.py input.pdf --review

Outputs:
    voters.csv
    voters.xlsx
    voters_review.csv  (when --review is used)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import fitz
import numpy as np
import pandas as pd
import pytesseract


COLUMNS = [
    "SR. NO",
    "VOTER_ID",
    "NAME",
    "FATHER'S NAME",
    "HOUSE NUMBER",
    "AGE",
    "GENDER",
]

DEV_TO_ASCII = str.maketrans("०१२३४५६७८९", "0123456789")


def clean(s: str) -> str:
    s = s.replace("\u200c", "").replace("\u200d", "")
    s = re.sub(r"\s+", " ", s)
    return s.strip(" \t|:-")


def norm_digits(s: str) -> str:
    return s.translate(DEV_TO_ASCII)


def render(page: fitz.Page, scale: int) -> np.ndarray:
    pix = page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        alpha=False,
        colorspace=fitz.csRGB,
    )
    return np.frombuffer(pix.samples, np.uint8).reshape(
        pix.height, pix.width, 3
    )


def grid_bounds(w: int, h: int):
    # Coordinates are fractions, so render scale can be changed.
    return (
        int(0.025 * w),
        int(0.980 * w),
        int(0.172 * h),
        int(0.935 * h),
    )


def get_card_crop(img: np.ndarray, card_no: int) -> np.ndarray:
    h, w = img.shape[:2]
    x0, x1, y0, y1 = grid_bounds(w, h)

    row = card_no // 3
    col = card_no % 3

    cw = (x1 - x0) / 3.0
    rh = (y1 - y0) / 9.0

    xa = int(x0 + col * cw) + 3
    xb = int(x0 + col * cw + 0.80 * cw)
    ya = int(y0 + row * rh) + 3
    yb = int(y0 + (row + 1) * rh) - 3

    return img[ya:yb, xa:xb]


def ocr_card(crop: np.ndarray) -> str:
    # PSM 11 separates Hindi text well and avoids the photo placeholder.
    gray = crop
    return pytesseract.image_to_string(
        gray,
        lang="hin",
        config="--oem 1 --psm 11",
    )


def clean_ocr_value(s: str) -> str:
    s = clean(s)
    s = re.sub(r"\b(Photo|photo|Available|is)\b.*$", "", s)
    # Common OCR garbage from the card photo/status area.
    s = s.strip(" .,:;-|[]()")
    s = re.sub(r"^[इपैहः]\s*[:：]\s*", "", s)
    return s


def parse_name_parent(ocr: str) -> tuple[str, str]:
    lines = [clean_ocr_value(x) for x in ocr.splitlines()]
    lines = [x for x in lines if x]

    name = ""
    parent = ""

    # 1) Best case: explicit "नाम:" line.
    for line in lines:
        if (
            "नाम:" in line
            and "का नाम:" not in line
            and not any(
                k in line for k in ["पिता", "पति", "माता", "अभिभावक"]
            )
        ):
            value = line.split("नाम:", 1)[1]
            value = clean_ocr_value(value)
            if value:
                name = value
                break

    # 2) Parent/guardian line.
    parent_keys = [
        "पिता का नाम:",
        "पति का नाम:",
        "माता का नाम:",
        "अभिभावक का नाम:",
    ]
    for line in lines:
        for key in parent_keys:
            if key in line:
                value = line.split(key, 1)[1]
                value = clean_ocr_value(value)
                if value:
                    parent = value
                    break
        # The OCR sometimes loses "पिता"/"पति" and leaves only "का नाम:".
        if not parent and "का नाम:" in line and not line.startswith("नाम:"):
            value = line.split("का नाम:", 1)[1]
            value = clean_ocr_value(value)
            if value:
                parent = value
        if parent:
            break

    # OCR fallback when "नाम:" is damaged/missing. Look for the first
    # meaningful Hindi line after the card's ID/serial line and before the
    # parent label.
    if not name:
        parent_i = next(
            (
                i for i, line in enumerate(lines)
                if (
                    ("पिता" in line or "पति" in line or "माता" in line)
                    and "नाम" in line
                )
                or "का नाम:" in line
            ),
            len(lines),
        )

        candidates = []
        for line in lines[:parent_i]:
            if any(k in line for k in [
                "मकान", "संख्या", "आयु", "लिंग", "Photo", "Available"
            ]):
                continue
            if re.search(r"[\u0900-\u097F]", line):
                # Remove obvious OCR punctuation/serial debris.
                candidate = re.sub(r"^[^ऀ-ॿ]+", "", line)
                candidate = clean_ocr_value(candidate)
                candidate = re.sub(r"^(?:नाम|नपम)?\s*[:：]\s*", "", candidate)

                # ID/serial OCR noise commonly contains digits. A real Hindi
                # name line in this layout does not normally contain digits.
                if re.search(r"\d", candidate):
                    continue
                if len(candidate) >= 2:
                    candidates.append(candidate)

        # The name is normally the last meaningful text line before the
        # parent-name line.
        if candidates:
            name = candidates[-1]

    # If a parent line was OCRed as "पिता का नाम:..." but split across
    # lines, take the line immediately after a parent-label-only line.
    if not parent:
        for i, line in enumerate(lines):
            if (
                ("पिता" in line or "पति" in line or "माता" in line)
                and "नाम" in line
                and not re.search(r"नाम\s*[:：]\s*.+", line)
            ):
                if i + 1 < len(lines):
                    candidate = clean_ocr_value(lines[i + 1])
                    if candidate:
                        parent = candidate
                        break

    return name, parent


def extract_text_layer_records(page_text: str) -> list[dict]:
    """
    The supplied PDFs have an embedded text layer whose Hindi labels are
    encoding-damaged, but the values (EPIC, age, gender, house) are still
    recoverable. Every voter block ends with its EPIC number.
    """
    lines = [clean(x) for x in page_text.splitlines()]
    id_positions = [
        i for i, line in enumerate(lines)
        if re.fullmatch(r"[A-Z]{2,5}\d{6,9}", line)
    ]

    records = []

    for pos in id_positions:
        # Fixed layout in this PDF:
        # age label/value, gender, house, name, parent, EPIC.
        age_line = lines[pos - 5] if pos >= 5 else ""
        gender_line = lines[pos - 4] if pos >= 4 else ""
        house_line = lines[pos - 3] if pos >= 3 else ""

        age_match = re.search(r"(\d{1,3})", norm_digits(age_line))
        age = age_match.group(1) if age_match else ""

        g = gender_line
        if g == "सर" or "सर" in g:
            gender = "स्त्री"
        elif g == "परष" or "परष" in g or "पुर" in g:
            gender = "पुरुष"
        else:
            gender = ""

        house = clean(house_line)
        # Remove obvious non-house words if the text layer has shifted.
        if not house:
            house = ""

        records.append({
            "VOTER_ID": lines[pos].upper(),
            "AGE": age,
            "GENDER": gender,
            "HOUSE NUMBER": house,
        })

    return records


def valid_data_page(page_text: str) -> bool:
    # Data pages have many EPIC IDs. Cover/map pages do not.
    return len(re.findall(r"\b[A-Z]{2,5}\d{6,9}\b", page_text.upper())) >= 3


def parse_pdf(pdf_path: Path, scale: int = 2) -> pd.DataFrame:
    doc = fitz.open(pdf_path)
    output = []
    serial = 1

    for page in doc:
        page_text = page.get_text("text")

        if not valid_data_page(page_text):
            continue

        text_records = extract_text_layer_records(page_text)
        img = render(page, scale)

        # The text layer gives us exact IDs and most numeric fields. OCR only
        # does the difficult Hindi name fields.
        for card_no, meta in enumerate(text_records[:27]):
            crop = get_card_crop(img, card_no)
            ocr = ocr_card(crop)
            name, parent = parse_name_parent(ocr)

            output.append({
                "SR. NO": str(serial),
                "VOTER_ID": meta["VOTER_ID"],
                "NAME": name,
                "FATHER'S NAME": parent,
                "HOUSE NUMBER": meta["HOUSE NUMBER"],
                "AGE": meta["AGE"],
                "GENDER": meta["GENDER"],
            })
            serial += 1

    doc.close()
    return pd.DataFrame(output, columns=COLUMNS)


def write_outputs(df: pd.DataFrame, base: Path, review: bool):
    base.parent.mkdir(parents=True, exist_ok=True)

    csv_path = base.with_suffix(".csv")
    xlsx_path = base.with_suffix(".xlsx")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Voters")
        ws = writer.book["Voters"]
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions

        widths = {
            "A": 10,
            "B": 18,
            "C": 30,
            "D": 30,
            "E": 18,
            "F": 10,
            "G": 12,
        }
        for col, width in widths.items():
            ws.column_dimensions[col].width = width

    review_path = None
    if review:
        review_df = df.copy()

        review_df["MISSING_FIELDS"] = review_df.apply(
            lambda r: ", ".join(
                col for col in COLUMNS if not str(r[col]).strip()
            ),
            axis=1,
        )

        review_df["DUPLICATE_VOTER_ID"] = (
            review_df["VOTER_ID"].ne("")
            & review_df["VOTER_ID"].duplicated(keep=False)
        )

        review_path = base.with_name(base.name + "_review.csv")
        review_df.to_csv(
            review_path,
            index=False,
            encoding="utf-8-sig",
        )

    return csv_path, xlsx_path, review_path


def main():
    ap = argparse.ArgumentParser(
        description="Extract voter data from Rajasthan voter-list PDFs."
    )
    ap.add_argument("pdf", type=Path)
    ap.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output basename, without extension",
    )
    ap.add_argument(
        "--scale",
        type=int,
        default=2,
        help="PDF render scale. 2 is a good speed/accuracy balance.",
    )
    ap.add_argument(
        "--review",
        action="store_true",
        help="Create an additional review CSV.",
    )
    args = ap.parse_args()

    if not args.pdf.exists():
        print(f"ERROR: PDF not found: {args.pdf}", file=sys.stderr)
        return 2

    try:
        languages = pytesseract.get_languages(config="")
    except Exception as exc:
        print(f"ERROR: Tesseract is not available: {exc}", file=sys.stderr)
        return 2

    if "hin" not in languages:
        print(
            "ERROR: Hindi OCR data is missing. Install tesseract-ocr-hin.",
            file=sys.stderr,
        )
        return 2

    base = args.output or args.pdf.with_name(
        args.pdf.stem + "_voters"
    )

    df = parse_pdf(args.pdf, scale=args.scale)
    csv_path, xlsx_path, review_path = write_outputs(
        df, base, args.review
    )

    print(f"Rows extracted: {len(df)}")
    print(f"CSV:   {csv_path}")
    print(f"Excel: {xlsx_path}")
    if review_path:
        print(f"Review: {review_path}")

    if not df.empty:
        print("\nPreview:")
        print(df.head(10).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
