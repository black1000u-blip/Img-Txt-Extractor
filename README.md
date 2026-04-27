# ImageGPT (OCR Question Extractor)

Minimal web UI that captures an image and extracts the first question + options using a local OCR model (PaddleOCR). No API keys.

## Run (local)

Requires Python 3.10+.

```
cd backend
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 3001
```

Open `http://localhost:3001`.

## Open on phone (same Wi‑Fi)

1) Keep the server running on your laptop.
2) Find your laptop IPv4 address (Windows: `ipconfig`).
3) On your phone, open: `http://<LAPTOP_IP>:3001`

Note: phone camera access (`getUserMedia`) usually requires HTTPS. If the camera doesn’t open over `http://<ip>`, use an HTTPS tunnel.

Example (ngrok):

```
ngrok http 3001
```

Open the `https://...` URL on your phone.

## Colab

Open `colab.ipynb` in Google Colab and run the cells. It installs dependencies, starts the server, and prints an HTTPS ngrok URL.

Colab one‑cell (quick run):

```python
!git clone https://github.com/black1000u-blip/imagegpt-ocr.git
%cd imagegpt-ocr
!pip -q install -r backend/requirements.txt

import threading, uvicorn
threading.Thread(
    target=lambda: uvicorn.run("backend.app:app", host="0.0.0.0", port=3001, log_level="info"),
    daemon=True,
).start()

from pyngrok import ngrok
import os
token = os.environ.get("NGROK_AUTHTOKEN", "").strip()
if token and token != "YOUR_TOKEN_HERE":
    ngrok.set_auth_token(token)
print("Open this on your phone:", ngrok.connect(3001, "http").public_url)
```

## Tips (phone)

- Use **Zoom** (bottom bar) to zoom before capture (hardware zoom if your device supports it).
- Use **Brightness** to help readability; capture also applies automatic text enhancement.
- Tap the captured image to open it and pinch-to-zoom.
