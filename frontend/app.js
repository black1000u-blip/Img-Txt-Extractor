const startBtn = document.getElementById("startBtn");
const captureBtn = document.getElementById("captureBtn");
const cameraEl = document.getElementById("camera");
const resultsEl = document.getElementById("results");
const resultTpl = document.getElementById("resultTpl");
const video = document.getElementById("video");
const statusEl = document.getElementById("status");
const imgDialog = document.getElementById("imgDialog");
const imgDialogImg = document.getElementById("imgDialogImg");
const zoomRange = document.getElementById("zoomRange");
const closeDialogBtn = document.getElementById("closeDialogBtn");
const brightRange = document.getElementById("brightRange");
const captureZoomRange = document.getElementById("captureZoomRange");
const imgDialogBody = document.querySelector(".imgDialogBody");

let stream = null;
let videoTrack = null;
let hasHardwareZoom = false;
let zoomApplyRaf = 0;
let imageCapture = null;

function setStatus(text) {
  statusEl.textContent = text || "";
}
// Hi
function stopStream() {
  if (!stream) return;
  for (const track of stream.getTracks()) track.stop();
  stream = null;
  videoTrack = null;
  hasHardwareZoom = false;
  imageCapture = null;
}

function clampByte(n) {
  return n < 0 ? 0 : n > 255 ? 255 : n | 0;
}

function enhanceCanvasForText(canvas) {
  const ctx = canvas.getContext("2d");
  const { width, height } = canvas;
  if (!ctx || width < 2 || height < 2) return;

  const img = ctx.getImageData(0, 0, width, height);
  const data = img.data;
  const pixels = width * height;

  // Luma histogram (0..255)
  const hist = new Uint32Array(256);
  const luma = new Uint8Array(pixels);

  for (let p = 0, i = 0; p < pixels; p++, i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const y = clampByte((r * 54 + g * 183 + b * 19) / 256); // ~0.2126,0.7152,0.0722
    luma[p] = y;
    hist[y]++;
  }

  // Contrast stretch by percentiles (reduces impact of glare/shadows)
  const lowCount = Math.floor(pixels * 0.01);
  const highCount = Math.floor(pixels * 0.99);
  let acc = 0;
  let low = 0;
  for (; low < 256; low++) {
    acc += hist[low];
    if (acc >= lowCount) break;
  }
  acc = 0;
  let high = 255;
  for (; high >= 0; high--) {
    acc += hist[high];
    if (pixels - acc <= highCount) break;
  }

  if (high <= low) {
    low = 0;
    high = 255;
  }

  const range = high - low || 1;

  // Gamma from mean (moves midtones for readability)
  let sum = 0;
  for (let p = 0; p < pixels; p++) {
    const v = clampByte(((luma[p] - low) * 255) / range);
    luma[p] = v;
    sum += v;
  }
  const mean = sum / pixels / 255;
  const target = 0.55;
  let gamma = mean > 0 ? Math.log(target) / Math.log(mean) : 1;
  gamma = Math.max(0.7, Math.min(1.5, gamma));

  // Apply gamma and write grayscale
  for (let p = 0, i = 0; p < pixels; p++, i += 4) {
    const v = clampByte(Math.pow(luma[p] / 255, gamma) * 255);
    luma[p] = v;
    data[i] = v;
    data[i + 1] = v;
    data[i + 2] = v;
  }

  // Light sharpen (helps small text); skip if huge to keep mobile fast
  if (pixels <= 2_500_000) {
    const out = new Uint8Array(pixels);
    for (let y = 1; y < height - 1; y++) {
      const row = y * width;
      for (let x = 1; x < width - 1; x++) {
        const idx = row + x;
        const c = luma[idx];
        const v = c * 5 - luma[idx - 1] - luma[idx + 1] - luma[idx - width] - luma[idx + width];
        out[idx] = clampByte(v);
      }
    }
    for (let x = 0; x < width; x++) {
      out[x] = luma[x];
      out[(height - 1) * width + x] = luma[(height - 1) * width + x];
    }
    for (let y = 0; y < height; y++) {
      out[y * width] = luma[y * width];
      out[y * width + (width - 1)] = luma[y * width + (width - 1)];
    }
    for (let p = 0, i = 0; p < pixels; p++, i += 4) {
      const v = out[p];
      data[i] = v;
      data[i + 1] = v;
      data[i + 2] = v;
    }
  }

  ctx.putImageData(img, 0, 0);
}

function addResultCard(dataUrl) {
  const node = resultTpl.content.cloneNode(true);
  const card = node.querySelector(".item");
  const photo = node.querySelector(".photo");
  const questionEl = node.querySelector(".question");
  const copyBtn = node.querySelector(".copyBtn");

  photo.src = dataUrl;
  questionEl.textContent = "Extracting...";

  photo.addEventListener("click", () => {
    if (imgDialog?.showModal) {
      imgDialogImg.src = dataUrl;
      zoomRange.value = "1";
      imgDialogImg.style.width = "100%";
      imgDialog.showModal();
      return;
    }
    window.open(dataUrl, "_blank");
  });

  copyBtn.addEventListener("click", async () => {
    const text = questionEl.textContent || "";
    if (!text || text === "Extracting...") return;
    try {
      await navigator.clipboard.writeText(text);
      setStatus("Copied.");
    } catch {
      setStatus("Copy failed.");
    }
  });

  resultsEl.append(card);
  window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" });
  return { questionEl, copyBtn };
}

function setBrightness(value) {
  const v = Number(value) || 1;
  video.style.filter = `brightness(${v})`;
}

brightRange?.addEventListener("input", (e) => setBrightness(e.target.value));
setBrightness(brightRange?.value || 1);

async function setupCameraControls(track) {
  if (!track?.getCapabilities) return;
  const caps = track.getCapabilities();

  // Best-effort continuous autofocus (supported on some mobile browsers/devices)
  if (Array.isArray(caps.focusMode) && caps.focusMode.includes("continuous")) {
    await track.applyConstraints({ advanced: [{ focusMode: "continuous" }] }).catch(() => {});
  }

  // Prefer hardware zoom (better clarity than digital zoom) when available
  if (caps.zoom && typeof caps.zoom.min === "number" && typeof caps.zoom.max === "number") {
    hasHardwareZoom = true;
    captureZoomRange.min = String(caps.zoom.min);
    captureZoomRange.max = String(caps.zoom.max);
    captureZoomRange.step = String(caps.zoom.step || 0.1);
    const current = track.getSettings?.().zoom;
    if (typeof current === "number") captureZoomRange.value = String(current);
    setCaptureZoom(captureZoomRange.value);
  } else {
    hasHardwareZoom = false;
    captureZoomRange.min = "1";
    captureZoomRange.max = "3";
    captureZoomRange.step = "0.1";
    if (!captureZoomRange.value) captureZoomRange.value = "1";
    setCaptureZoom(captureZoomRange.value);
  }

  // Create ImageCapture (high-res still capture) when available
  if (window.ImageCapture) {
    imageCapture = new ImageCapture(track);
  } else {
    imageCapture = null;
  }
}

function setCaptureZoom(value) {
  const z = Number(value) || 1;
  if (hasHardwareZoom && videoTrack?.applyConstraints) {
    cancelAnimationFrame(zoomApplyRaf);
    zoomApplyRaf = requestAnimationFrame(() => {
      videoTrack
        .applyConstraints({ advanced: [{ zoom: z }] })
        .catch(() => {});
    });
    video.style.transform = "scale(1)";
    return;
  }
  video.style.transform = `scale(${z})`;
}

captureZoomRange?.addEventListener("input", (e) => setCaptureZoom(e.target.value));
setCaptureZoom(captureZoomRange?.value || 1);

function setDialogZoom(value) {
  const z = Math.max(1, Math.min(3, Number(value) || 1));
  zoomRange.value = String(z);
  imgDialogImg.style.width = `${z * 100}%`;
}

zoomRange?.addEventListener("input", (e) => {
  setDialogZoom(e.target.value);
});

closeDialogBtn?.addEventListener("click", () => imgDialog.close());
imgDialog?.addEventListener("click", (e) => {
  if (e.target === imgDialog) imgDialog.close();
});
imgDialog?.addEventListener("close", () => (imgDialogImg.src = ""));

// Pinch-to-zoom inside the image dialog (mobile)
const pointers = new Map();
let pinchStartDist = 0;
let pinchStartZoom = 1;

function dist(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

imgDialogBody?.addEventListener("pointerdown", (e) => {
  if (!imgDialog?.open) return;
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  imgDialogBody.setPointerCapture?.(e.pointerId);
  if (pointers.size === 2) {
    const [p1, p2] = Array.from(pointers.values());
    pinchStartDist = dist(p1, p2) || 1;
    pinchStartZoom = Number(zoomRange.value) || 1;
  }
});

imgDialogBody?.addEventListener("pointermove", (e) => {
  if (!imgDialog?.open) return;
  if (!pointers.has(e.pointerId)) return;
  pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
  if (pointers.size !== 2) return;
  const [p1, p2] = Array.from(pointers.values());
  const d = dist(p1, p2) || 1;
  const next = pinchStartZoom * (d / pinchStartDist);
  setDialogZoom(next);
});

function clearPointer(e) {
  pointers.delete(e.pointerId);
  if (pointers.size < 2) pinchStartDist = 0;
}

imgDialogBody?.addEventListener("pointerup", clearPointer);
imgDialogBody?.addEventListener("pointercancel", clearPointer);

startBtn.addEventListener("click", async () => {
  setStatus("");

  if (!window.isSecureContext) {
    setStatus("Camera needs HTTPS on mobile. Open the app via an https URL (e.g. ngrok).");
    return;
  }

  try {
    cameraEl.classList.remove("hidden");
    cameraEl.classList.add("active");
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1920 },
        height: { ideal: 1080 }
      }
    });
    videoTrack = stream.getVideoTracks?.()[0] || null;
    await setupCameraControls(videoTrack);
    video.srcObject = stream;
    await new Promise((resolve) => {
      if (video.readyState >= 1) return resolve();
      video.onloadedmetadata = () => resolve();
    });
    await video.play().catch(() => {});
    setBrightness(brightRange?.value || 1);
    setCaptureZoom(captureZoomRange?.value || 1);
    captureBtn.disabled = false;
    startBtn.disabled = true;
  } catch (err) {
    cameraEl.classList.add("hidden");
    cameraEl.classList.remove("active");
    const name = err?.name ? String(err.name) : "Error";
    const msg = err?.message ? String(err.message) : "Camera error.";
    setStatus(`${name}: ${msg}`);
  }
});

captureBtn.addEventListener("click", async () => {
  if (!stream) return;
  setStatus("Extracting...");
  captureBtn.disabled = true;

  let source = video;
  let vw = video.videoWidth || 1280;
  let vh = video.videoHeight || 720;

  if (imageCapture?.takePhoto) {
    try {
      const blob = await imageCapture.takePhoto();
      if (window.createImageBitmap) {
        const bmp = await createImageBitmap(blob);
        source = bmp;
        vw = bmp.width;
        vh = bmp.height;
      } else {
        const url = URL.createObjectURL(blob);
        const img = new Image();
        await new Promise((resolve, reject) => {
          img.onload = () => resolve();
          img.onerror = () => reject(new Error("Image load failed"));
          img.src = url;
        });
        URL.revokeObjectURL(url);
        source = img;
        vw = img.naturalWidth || vw;
        vh = img.naturalHeight || vh;
      }
    } catch {
      // fall back to video frame
    }
  }

  const maxDim = 1600;
  const scale = Math.min(1, maxDim / Math.max(vw, vh));

  const canvas = document.createElement("canvas");
  canvas.width = Math.max(2, Math.floor(vw * scale));
  canvas.height = Math.max(2, Math.floor(vh * scale));
  const ctx = canvas.getContext("2d");
  const b = Number(brightRange?.value || 1);
  const z = hasHardwareZoom ? 1 : Number(captureZoomRange?.value || 1);
  try {
    ctx.filter = `brightness(${b})`;
  } catch {}

  const sw = Math.max(1, Math.floor(vw / z));
  const sh = Math.max(1, Math.floor(vh / z));
  const sx = Math.max(0, Math.floor((vw - sw) / 2));
  const sy = Math.max(0, Math.floor((vh - sh) / 2));
  ctx.drawImage(source, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);

  // Auto enhance for text: contrast stretch + gamma + light sharpen
  enhanceCanvasForText(canvas);

  const dataUrl = canvas.toDataURL("image/jpeg", 0.9);
  const base64 = dataUrl.split(",")[1];

  stopStream();
  cameraEl.classList.add("hidden");
  cameraEl.classList.remove("active");
  startBtn.disabled = false;
  try {
    if (source?.close && source !== video) source.close();
  } catch {}

  const { questionEl, copyBtn } = addResultCard(dataUrl);

  try {
    const res = await fetch("/extract", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: base64 })
    });

    const rawText = await res.text().catch(() => "");
    const data = rawText ? JSON.parse(rawText) : {};
    if (!res.ok) {
      const msg = data?.details || data?.error || rawText || `Request failed (HTTP ${res.status}).`;
      throw new Error(msg);
    }

    const q = String(data?.question || "").trim();
    questionEl.textContent = q || "(No question found)";
    copyBtn.disabled = !q;
    setStatus("");
  } catch (err) {
    questionEl.textContent = "(Extraction error)";
    setStatus(err?.message ? `Extraction error: ${err.message}` : "Extraction error.");
  }
});

window.addEventListener("beforeunload", stopStream);
