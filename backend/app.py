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
  raw = base64.b64decode(s, validate=False)
  img = Image.open(io.BytesIO(raw)).convert("RGB")
  img = ImageOps.exif_transpose(img)
  return img


def enhance_for_text(img: Image.Image) -> Image.Image:
  img = ImageOps.autocontrast(img, cutoff=2)
  img = ImageEnhance.Contrast(img).enhance(1.35)
  img = ImageEnhance.Sharpness(img).enhance(1.4)
  return img


def format_options_each_line(text: str) -> str:
  t = (text or "").strip()
  if not t:
    return ""

  first_opt = re.search(r"\bA\)\s*", t)
  if not first_opt:
    return t

  q = t[: first_opt.start()].strip().rstrip("-•").strip()
  opts = t[first_opt.start() :].strip()
  opts = re.sub(r"^[-•]\s*", "", opts)
  opts = re.sub(r"\s*[-•]?\s*\b([A-D])\)\s*", r"\n\1) ", opts)
  opts = "\n".join([line.strip() for line in opts.splitlines() if line.strip()])
  return "\n".join([p for p in [q, opts] if p])


def ocr_lines(img: Image.Image) -> List[Tuple[float, float, float, str, float]]:
  """
  Returns lines: (y, x, h, text, score) sorted top-to-bottom, left-to-right.
  """
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
  return lines


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


def extract_first_question(text_lines: List[str]) -> str:
  if not text_lines:
    return ""

  joined = "\n".join(text_lines)
  joined = re.sub(r"^I['’]?\s*m\s*sorry[\s\S]*?\n+", "", joined, flags=re.I).strip()
  joined = re.sub(r"^Sorry[\s\S]*?\n+", "", joined, flags=re.I).strip()

  # Find first plausible question start
  start_idx = 0
  for i, line in enumerate(text_lines):
    if re.search(r"\bA\)\b", line) or re.search(r"[?]$", line) or re.match(r"^\s*(Q\s*\d+|\d+)[\).:]", line, re.I):
      start_idx = i
      break

  collected = []
  seen_a = False
  seen_d = False

  for i in range(start_idx, len(text_lines)):
    line = text_lines[i].strip()
    if not line:
      continue

    if collected and re.match(r"^\s*(Answer|Solution)\s*:", line, re.I):
      break

    if collected and re.match(r"^\s*(Q\s*\d+|\d+)[\).:]", line, re.I) and not re.match(r"^\s*A\)", line):
      break

    if re.search(r"\bA\)\b", line):
      seen_a = True
    if re.search(r"\bD\)\b", line):
      seen_d = True

    collected.append(line)

    if seen_a and seen_d:
      # Stop after options complete unless next line looks like more options/continuation.
      if i + 1 < len(text_lines) and not re.match(r"^\s*[A-D]\)", text_lines[i + 1].strip()):
        break

  return format_options_each_line("\n".join(collected).strip())


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
    lines = ocr_lines(img)
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
