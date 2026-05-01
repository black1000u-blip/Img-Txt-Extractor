import base64
import io
import os
import re
import traceback
from typing import Any, List, Tuple

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image, ImageEnhance, ImageOps
from starlette.concurrency import run_in_threadpool

_ocr = None


def get_ocr():
    global _ocr
    if _ocr is None:
        from paddleocr import PaddleOCR
        _ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    return _ocr


def decode_base64_image(image_b64: str) -> Image.Image:
    s = image_b64.strip()
    if s.startswith("data:"):
        s = s.split(",", 1)[-1]
    try:
        raw = base64.b64decode(s, validate=True)
    except Exception:
        raw = base64.b64decode(s, validate=False)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img = ImageOps.exif_transpose(img)
    return img


def enhance_for_text(img: Image.Image) -> Image.Image:
    img = ImageOps.autocontrast(img, cutoff=2)
    img = ImageEnhance.Contrast(img).enhance(1.35)
    img = ImageEnhance.Sharpness(img).enhance(1.4)
    return img


def ocr_lines(img: Image.Image) -> Tuple[List[Tuple[float, float, float, str, float]], int, int]:
    """Returns (lines, img_w, img_h). lines: (y, x, h, text, score)."""
    img_w, img_h = img.size
    arr = np.array(img)
    raw = get_ocr().ocr(arr, cls=True) or []

    def is_line_item(x: Any) -> bool:
        return (
            isinstance(x, list)
            and len(x) == 2
            and isinstance(x[0], list)
            and isinstance(x[1], (list, tuple))
            and len(x[1]) >= 1
            and isinstance(x[1][0], str)
        )

    if raw and is_line_item(raw[0]):
        result = raw
    elif raw and isinstance(raw[0], list) and raw[0] and is_line_item(raw[0][0]):
        result = raw[0]
    else:
        result = []

    lines = []
    for item in result:
        if not item or len(item) < 2:
            continue
        box, meta = item[0], item[1]
        text = str(meta[0] or "").strip()
        score = float(meta[1] or 0.0)
        if not text:
            continue
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x = float(min(xs))
        y = float(min(ys))
        h = float(max(ys) - min(ys))
        lines.append((y, x, h, text, score))

    lines.sort(key=lambda t: (t[0], t[1]))
    return lines, img_w, img_h


def filter_to_main_content(
    lines: List[Tuple[float, float, float, str, float]],
    img_w: int,
    img_h: int,
) -> List[Tuple[float, float, float, str, float]]:
    """
    Removes left sidebar (navigation) and right sidebar text.
    Anchors the content region using question-1 marker and option lines.
    """
    if not lines or img_w == 0:
        return lines

    # --- Find left boundary using "1." or "1)" anchor ---
    left_bound = None
    for y, x, h, text, score in lines:
        if re.match(r"^\s*1\s*[.)]\s*\S", text):
            left_bound = max(0.0, x - img_w * 0.03)
            break

    # Fallback: use first A) option x-position
    if left_bound is None:
        for y, x, h, text, score in lines:
            if re.match(r"^\s*[A-D]\)\s*\S", text):
                left_bound = max(0.0, x - img_w * 0.03)
                break

    # Final fallback: 12% from left
    if left_bound is None:
        left_bound = img_w * 0.12

    # --- Find right boundary ---
    # For landscape images, right sidebar typically starts at ~70% of width
    if img_w > img_h:
        right_bound = img_w * 0.72
    else:
        right_bound = float(img_w)

    filtered = [
        (y, x, h, text, score)
        for y, x, h, text, score in lines
        if x >= left_bound and x <= right_bound
    ]
    return filtered if filtered else lines


def merge_lines(lines: List[Tuple[float, float, float, str, float]]) -> List[str]:
    if not lines:
        return []

    heights = sorted([h for _, _, h, _, _ in lines if h > 0])
    median_h = heights[len(heights) // 2] if heights else 16.0
    threshold = max(8.0, median_h * 0.6)

    merged: List[List[Tuple[float, float, str]]] = []
    cur: List[Tuple[float, float, str]] = []
    cur_y = None

    for y, x, h, text, _score in lines:
        if cur_y is None or abs(y - cur_y) <= threshold:
            cur.append((y, x, text))
            cur_y = y if cur_y is None else (cur_y * 0.7 + y * 0.3)
            continue
        merged.append(cur)
        cur = [(y, x, text)]
        cur_y = y

    if cur:
        merged.append(cur)

    out = []
    for group in merged:
        group.sort(key=lambda t: t[1])
        out.append(" ".join([t[2] for t in group]).strip())
    return [s for s in out if s]


def format_options_each_line(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""

    # Strip leading question number  e.g. "1." or "1)"
    t = re.sub(r"^\s*[Qq]?\s*\d+\s*[.)]\s*", "", t).strip()

    first_opt = re.search(r"\bA\)\s*", t)
    if not first_opt:
        return t

    q = t[: first_opt.start()].strip().rstrip("-•").strip()
    opts = t[first_opt.start():].strip()
    opts = re.sub(r"^[-•]\s*", "", opts)
    opts = re.sub(r"\s*[-•]?\s*\b([A-D])\)\s*", r"\n\1) ", opts)
    opts = "\n".join([line.strip() for line in opts.splitlines() if line.strip()])
    return "\n".join([p for p in [q, opts] if p])


def extract_first_question(text_lines: List[str]) -> str:
    if not text_lines:
        return ""

    # --- Find where Question 1 starts ---
    q1_idx = None
    for i, line in enumerate(text_lines):
        if re.match(r"^\s*1\s*[.)]\s*\S", line):
            q1_idx = i
            break

    # Fallback: first line with A) option — back up 2 lines for question body
    if q1_idx is None:
        for i, line in enumerate(text_lines):
            if re.search(r"\bA\)\s*\S", line):
                q1_idx = max(0, i - 2)
                break

    # Last fallback: first line with a question mark at end
    if q1_idx is None:
        for i, line in enumerate(text_lines):
            if re.search(r"\?\s*$", line.strip()) and len(line.strip()) > 15:
                q1_idx = i
                break

    if q1_idx is None:
        q1_idx = 0

    collected = []
    seen_options: set = set()

    for i in range(q1_idx, len(text_lines)):
        line = text_lines[i].strip()
        if not line:
            continue

        # Stop when question 2 begins
        if collected and re.match(r"^\s*2\s*[.)]\s*\S", line):
            break

        # Stop at answer/solution marker
        if collected and re.match(r"^\s*(Answer|Solution|Explanation|Ans)\s*[:.)]", line, re.I):
            break

        # Track options seen
        for opt in "ABCD":
            if re.search(rf"\b{opt}\)\s*\S", line):
                seen_options.add(opt)

        collected.append(line)

        # Stop once D) option is collected
        if "D" in seen_options:
            break

    if not collected:
        return ""

    joined = "\n".join(collected) if len(collected) > 1 else collected[0]
    return format_options_each_line(joined)


class ExtractIn(BaseModel):
    image: str


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/extract")
async def extract(payload: ExtractIn) -> Any:
    def run() -> str:
        img = decode_base64_image(payload.image)
        img = enhance_for_text(img)
        lines, img_w, img_h = ocr_lines(img)
        lines = filter_to_main_content(lines, img_w, img_h)
        merged = merge_lines(lines)
        return extract_first_question(merged)

    try:
        question = await run_in_threadpool(run)
        return {"question": question or ""}
    except Exception as e:
        details = "".join(traceback.format_exception_only(type(e), e)).strip()
        return JSONResponse(status_code=500, content={"error": "OCR failed.", "details": details})


FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")