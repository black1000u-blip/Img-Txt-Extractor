import base64
import io
import os
import re
import traceback
from typing import Any, List, Tuple, Optional

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image, ImageEnhance, ImageOps
from starlette.concurrency import run_in_threadpool

_ocr = None

# ── Keyboard / UI noise patterns to discard ───────────────────────────────
_NOISE = re.compile(
    r"^("
    r"F\d{1,2}"                          # F1-F12
    r"|Pause|Scr[lL]k?|Prtsc?|Prise"    # keyboard labels
    r"|Ins|Del|Home|End|PgUp|PgDn"
    r"|NumLk|SysRq|Break|CapsLock"
    r"|Shift|Ctrl|Alt|Tab|Esc|Enter"
    r"|RF|[A-Z]{1,3}\d*"                 # random short keys like RF, K, U
    r")\s*$",
    re.IGNORECASE,
)

_BROWSER_URL = re.compile(r"https?://|www\.", re.IGNORECASE)


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
    img_w, img_h = img.size
    arr = np.array(img)
    raw = get_ocr().ocr(arr, cls=True) or []

    def is_line_item(x: Any) -> bool:
        return (
            isinstance(x, list) and len(x) == 2
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
        lines.append((float(min(ys)), float(min(xs)), float(max(ys) - min(ys)), text, score))

    lines.sort(key=lambda t: (t[0], t[1]))
    return lines, img_w, img_h


def remove_noise(
    lines: List[Tuple[float, float, float, str, float]]
) -> List[Tuple[float, float, float, str, float]]:
    """Drop keyboard keys, URLs, single-char garbage, very short tokens."""
    clean = []
    for y, x, h, text, score in lines:
        t = text.strip()
        if len(t) <= 1:
            continue
        if _NOISE.match(t):
            continue
        if _BROWSER_URL.search(t):
            continue
        clean.append((y, x, h, text, score))
    return clean


def remove_sidebars(
    lines: List[Tuple[float, float, float, str, float]],
    img_w: int,
    img_h: int,
) -> List[Tuple[float, float, float, str, float]]:
    """
    Crop left and right sidebars using the first question-number line as
    the left anchor. For landscape images also crop the right sidebar zone.
    """
    if not lines or img_w == 0:
        return lines

    # Find left anchor: first line that looks like a question number "N." or "N)"
    left_bound = None
    for y, x, h, text, score in lines:
        if re.match(r"^\s*\d+\s*[.)]\s*\S", text):
            left_bound = max(0.0, x - img_w * 0.04)
            break

    # Fallback: first option line A)
    if left_bound is None:
        for y, x, h, text, score in lines:
            if re.match(r"^\s*[A-D]\)\s*\S", text):
                left_bound = max(0.0, x - img_w * 0.04)
                break

    if left_bound is None:
        left_bound = img_w * 0.10   # generic 10% trim

    # Right boundary: landscape images usually have a right sidebar
    right_bound = img_w * 0.72 if img_w > img_h else float(img_w)

    filtered = [
        item for item in lines
        if item[1] >= left_bound and item[1] <= right_bound
    ]
    return filtered if filtered else lines


def merge_into_text_lines(
    lines: List[Tuple[float, float, float, str, float]]
) -> List[str]:
    """Group nearby y-rows into single text lines, sorted left-to-right."""
    if not lines:
        return []

    heights = sorted([h for _, _, h, _, _ in lines if h > 0])
    median_h = heights[len(heights) // 2] if heights else 16.0
    threshold = max(8.0, median_h * 0.6)

    groups: List[List[Tuple[float, float, str]]] = []
    cur: List[Tuple[float, float, str]] = []
    cur_y: Optional[float] = None

    for y, x, h, text, _ in lines:
        if cur_y is None or abs(y - cur_y) <= threshold:
            cur.append((y, x, text))
            cur_y = y if cur_y is None else cur_y * 0.7 + y * 0.3
        else:
            groups.append(cur)
            cur = [(y, x, text)]
            cur_y = y

    if cur:
        groups.append(cur)

    result = []
    for g in groups:
        g.sort(key=lambda t: t[1])
        result.append(" ".join(t[2] for t in g).strip())
    return [s for s in result if s]


def format_clean_output(text: str) -> str:
    """
    Strip question number prefix, then format options A-D each on its own line.
    """
    t = text.strip()
    if not t:
        return ""

    # Remove leading question number like "2." / "3)" / "Q2."
    t = re.sub(r"^\s*[Qq]?\s*\d+\s*[.)]\s*", "", t).strip()

    # Find first option marker
    m = re.search(r"\bA\)\s*", t)
    if not m:
        return t

    question = t[: m.start()].strip().rstrip("-•").strip()
    opts_raw = t[m.start():]
    # Split on A) B) C) D) boundaries
    opts_raw = re.sub(r"\s*\b([A-D])\)\s*", r"\n\1) ", opts_raw)
    opts_lines = [ln.strip() for ln in opts_raw.splitlines() if ln.strip()]
    opts = "\n".join(opts_lines)

    parts = [p for p in [question, opts] if p]
    return "\n".join(parts)


def extract_top_question(text_lines: List[str]) -> str:
    """
    Find the FIRST (topmost) question visible — any question number.
    Collect it and its A-D options. Stop at the next question or after D).
    """
    if not text_lines:
        return ""

    # ── Step 1: find the topmost question number line ─────────────────────
    q_idx = None
    q_num = None
    for i, line in enumerate(text_lines):
        m = re.match(r"^\s*(\d+)\s*[.)]\s*\S", line)
        if m:
            q_idx = i
            q_num = int(m.group(1))
            break

    # Fallback A: a line that ends with "?" (question body without number)
    if q_idx is None:
        for i, line in enumerate(text_lines):
            if re.search(r"\?\s*$", line) and len(line.strip()) > 20:
                q_idx = i
                break

    # Fallback B: wherever A) first appears, back up 3 lines
    if q_idx is None:
        for i, line in enumerate(text_lines):
            if re.search(r"\bA\)\s*\S", line):
                q_idx = max(0, i - 3)
                break

    if q_idx is None:
        q_idx = 0

    next_q_pat = (
        re.compile(rf"^\s*{q_num + 1}\s*[.)]\s*\S") if q_num is not None
        else re.compile(r"^\s*\d+\s*[.)]\s*\S")
    )

    # ── Step 2: collect lines for this question ───────────────────────────
    collected: List[str] = []
    seen_opts: set = set()

    for i in range(q_idx, len(text_lines)):
        line = text_lines[i].strip()
        if not line:
            continue

        # Stop at the next question number
        if collected and next_q_pat.match(line):
            break

        # Stop at answer/solution/explanation markers
        if collected and re.match(
            r"^\s*(Answer|Solution|Explanation|Ans)\s*[:.)]\s*", line, re.I
        ):
            break

        # Track which options we've seen
        for opt in "ABCD":
            if re.search(rf"\b{opt}\)\s*\S", line):
                seen_opts.add(opt)

        collected.append(line)

        # Done once option D) is collected
        if "D" in seen_opts:
            break

    if not collected:
        return ""

    return format_clean_output("\n".join(collected))


# ── FastAPI app ───────────────────────────────────────────────────────────

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
        lines = remove_noise(lines)
        lines = remove_sidebars(lines, img_w, img_h)
        text_lines = merge_into_text_lines(lines)
        return extract_top_question(text_lines)

    try:
        question = await run_in_threadpool(run)
        return {"question": question or ""}
    except Exception as e:
        details = "".join(traceback.format_exception_only(type(e), e)).strip()
        return JSONResponse(
            status_code=500,
            content={"error": "OCR failed.", "details": details},
        )


FRONTEND_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "frontend")
)
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")