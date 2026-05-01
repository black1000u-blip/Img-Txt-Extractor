import os, sys, subprocess, threading, time, traceback, urllib.request, urllib.error

REPO  = "https://github.com/black1000u-blip/Img-Txt-Extractor.git"
DIR   = "Img-Txt-Extractor"
PORT  = 3001
TOKEN = "3AbFUwuUiWdVY3KG97gvOnqiR18_7S56jZJzefkv21hPXGqar"

print("=" * 55)
print("  Step 1: Clone / update repo")
print("=" * 55)

if os.path.isdir(DIR):
    print(f"'{DIR}' already exists — pulling latest...")
    r = subprocess.run(["git", "-C", DIR, "pull"], capture_output=True, text=True)
    print(r.stdout or "(up to date)")
    if r.stderr:
        print("GIT STDERR:", r.stderr)
else:
    r = subprocess.run(["git", "clone", REPO], capture_output=True, text=True)
    print(r.stdout or "")
    if r.returncode != 0:
        print("GIT CLONE ERROR:\n", r.stderr)
        raise RuntimeError("Clone failed.")
    print("Clone OK.")

os.chdir(DIR)
print("Working dir:", os.getcwd())

print("\n" + "=" * 55)
print("  Step 2: Install dependencies")
print("=" * 55)

r2 = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-r", "backend/requirements.txt"],
    capture_output=False,
    text=True
)
if r2.returncode != 0:
    raise RuntimeError("pip install failed — see output above.")
print("\nInstall complete.")

print("\n" + "=" * 55)
print("  Step 3: Start FastAPI server")
print("=" * 55)

_crash_log = []

def _run_server():
    try:
        import uvicorn
        uvicorn.run(
            "backend.app:app",
            host="0.0.0.0",
            port=PORT,
            log_level="debug",
            access_log=True,
        )
    except Exception:
        tb = traceback.format_exc()
        _crash_log.append(tb)
        print("\n[SERVER CRASH]\n" + tb, flush=True)

threading.Thread(target=_run_server, daemon=True).start()
print(f"Server thread started, waiting 10 s for PaddleOCR init ...")
time.sleep(10)

if _crash_log:
    raise RuntimeError("Server crashed on startup — see crash log above.")

print("\n" + "=" * 55)
print("  Step 4: Health check")
print("=" * 55)

alive = False
for attempt in range(1, 16):
    try:
        urllib.request.urlopen(f"http://localhost:{PORT}", timeout=3)
        alive = True
        break
    except urllib.error.HTTPError:
        alive = True
        break
    except Exception as ex:
        print(f"  [{attempt}/15] not ready: {ex}")
        time.sleep(2)

if not alive:
    raise RuntimeError("Server never responded. Check crash output above.")

print("Server is responding OK.")

print("\n" + "=" * 55)
print("  Step 5: Start ngrok tunnel")
print("=" * 55)

try:
    from pyngrok import ngrok
    ngrok.set_auth_token(TOKEN)
    tunnel = ngrok.connect(PORT, "http")
    url = tunnel.public_url
    if url.startswith("http://"):
        url = "https://" + url[7:]
    print("\n" + "=" * 55)
    print("  APP IS LIVE:")
    print(f"  {url}")
    print("  Open on your phone — camera requires HTTPS")
    print("=" * 55)
except Exception:
    print("[NGROK ERROR]")
    traceback.print_exc()
    raise

print("\nLive — server logs will print below. Press Stop to exit.\n")
try:
    while True:
        time.sleep(2)
        if _crash_log:
            print("\n[LATE SERVER ERROR DETECTED]")
            for entry in _crash_log:
                print(entry)
            break
except KeyboardInterrupt:
    print("Stopped by user.")
