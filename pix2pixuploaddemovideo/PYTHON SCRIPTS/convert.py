#!/usr/bin/env python3
"""
Run an ml5 pix2pix .pict model over an image sequence, GIF or video, locally.

Setup:
    pip install playwright pillow opencv-python numpy tqdm numba
    playwright install chromium
    (ffmpeg on PATH is optional; falls back to OpenCV's mp4v writer)

Usage:
    python pix2pix_transfer.py --model model.pict --input frames_folder/
    python pix2pix_transfer.py --model model.pict --input clip.mp4 --frame-step 2
    python pix2pix_transfer.py --model model.pict --input anim.gif --out result.mp4
"""
import argparse, atexit, base64, functools, http.server, io, re, shutil, subprocess
import sys, tempfile, threading, time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageSequence
from tqdm import tqdm
from playwright.sync_api import sync_playwright

SIZE, DISPLAY = 256, 512
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
VID_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

HTML = """<!doctype html><html><body>
<script src="https://unpkg.com/ml5@0.12.2/dist/ml5.min.js"></script>
<script>
window.ready = new Promise(res => { window.p2p = ml5.pix2pix('model.pict', () => res(true)); });
window.run = b64 => new Promise((resolve, reject) => {
  const img = new Image();
  img.onload = () => {
    // one persistent on-page canvas, reused every frame (like the web app's modelCanvas)
    if (!window.cv) { window.cv = document.createElement('canvas'); cv.width = cv.height = 256; document.body.appendChild(cv); }
    const c = window.cv, ctx = c.getContext('2d');
    ctx.clearRect(0, 0, 256, 256); ctx.drawImage(img, 0, 0, 256, 256);
    p2p.transfer(c, async (err, res) => {
      if (err) return reject((err && err.stack) || String(err));
      let src = res.src;
      if (src.startsWith('blob:')) {
        const b = await (await fetch(src)).blob();
        src = await new Promise(r => { const f = new FileReader(); f.onload = () => r(f.result); f.readAsDataURL(b); });
      }
      resolve(src);
    });
  };
  img.src = 'data:image/png;base64,' + b64;
});
window.glRenderer = () => {
  const gl = document.createElement('canvas').getContext('webgl');
  if (!gl) return 'no WebGL';
  const e = gl.getExtension('WEBGL_debug_renderer_info');
  return e ? gl.getParameter(e.UNMASKED_RENDERER_WEBGL) : 'unknown';
};
</script></body></html>"""


# ---------- dithering ----------
def _fs_dither(a, levels):
    h, w, _ = a.shape
    step = 255.0 / (levels - 1)
    for y in range(h):
        for x in range(w):
            for c in range(3):
                old = a[y, x, c]
                new = np.floor(np.floor(old / step + 0.5) * step + 0.5)
                a[y, x, c] = new
                e = old - new
                if x + 1 < w:
                    a[y, x+1, c] = min(255.0, max(0.0, a[y, x+1, c] + e*7/16))
                if y + 1 < h:
                    if x > 0:
                        a[y+1, x-1, c] = min(255.0, max(0.0, a[y+1, x-1, c] + e*3/16))
                    a[y+1, x, c] = min(255.0, max(0.0, a[y+1, x, c] + e*5/16))
                    if x + 1 < w:
                        a[y+1, x+1, c] = min(255.0, max(0.0, a[y+1, x+1, c] + e*1/16))

try:
    from numba import njit
    fs_dither = njit(cache=True)(_fs_dither)
except ImportError:
    fs_dither = _fs_dither
    print("note: numba not installed, dithering will be slow (pip install numba)")


# ---------- input handling ----------
natural = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]

def to_model_frame(img):
    img = img.convert("RGBA")
    bg = Image.new("RGBA", img.size, (0, 0, 0, 255))
    bg.alpha_composite(img)
    return bg.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR)  # stretched, like the web app

def prepare_frames(inp, outdir, step, max_frames, fps):
    """Write 256x256 PNGs to outdir. Returns (count, fps)."""
    inp = Path(inp)
    n = 0
    if inp.is_file() and inp.suffix.lower() in VID_EXT:
        cap = cv2.VideoCapture(str(inp))
        fps = fps or (cap.get(cv2.CAP_PROP_FPS) or 24) / step
        i = 0
        while n < max_frames:
            ok, fr = cap.read()
            if not ok: break
            if i % step == 0:
                to_model_frame(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))).save(outdir / f"{n:05d}.png")
                n += 1
            i += 1
        cap.release()
        return n, fps

    paths = [inp] if inp.is_file() else sorted(
        (p for p in inp.rglob("*") if p.suffix.lower() in IMG_EXT and not p.name.startswith(".")),
        key=lambda p: natural(p.relative_to(inp)))
    for p in paths:
        im = Image.open(p)
        if getattr(im, "n_frames", 1) > 1:  # animated gif / webp
            if len(paths) == 1 and not fps and im.info.get("duration"):
                fps = 1000 / im.info["duration"] / step
            for j, fr in enumerate(ImageSequence.Iterator(im)):
                if j % step == 0 and n < max_frames:
                    to_model_frame(fr.copy()).save(outdir / f"{n:05d}.png"); n += 1
        elif n < max_frames:
            to_model_frame(im).save(outdir / f"{n:05d}.png"); n += 1
    return n, fps or 8


# ---------- model ----------
def run_model(model_path, frames_dir, raw_dir, headed, swiftshader, debug=False):
    web = Path(tempfile.mkdtemp(prefix="p2p_web_"))
    work = web
    atexit.register(shutil.rmtree, web, True)   # clean up even if the run crashes
    shutil.copy(model_path, work / "model.pict")
    (work / "index.html").write_text(HTML)

    class H(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a): pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(H, directory=str(work)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/index.html"

    args = ["--ignore-gpu-blocklist", "--enable-gpu-rasterization"]
    if swiftshader:
        args += ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"]

    files = sorted(frames_dir.glob("*.png"))
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not headed, args=args)

        def new_page():
            pg = browser.new_page()
            if debug:
                pg.on("console", lambda m: m.type == "error" and "Failed to load resource" not in m.text
                      and tqdm.write(f"[page error] {m.text[:300]}"))
                pg.on("pageerror", lambda e: tqdm.write(f"[page error] {str(e)[:300]}"))
            pg.goto(url)
            pg.evaluate("window.ready")
            return pg

        print("loading model...", flush=True)
        t0 = time.time()
        pg = new_page()
        print(f"model loaded in {time.time()-t0:.0f}s | WebGL renderer: {pg.evaluate('window.glRenderer()')}")
        if "swiftshader" in pg.evaluate("window.glRenderer()").lower():
            print("WebGL is running on the CPU (SwiftShader), so this will be slow. Try --headed for GPU.")

        calls, reload_every = 0, 0   # transfers done on the current page / proactive reload interval
        for i, f in enumerate(tqdm(files, unit="frame", ncols=100)):
            if (raw_dir / f.name).exists():      # already done in a previous run
                continue
            b64 = base64.b64encode(f.read_bytes()).decode()
            if reload_every and calls >= reload_every:
                pg.close(); pg = new_page(); calls = 0
            for _ in range(3):
                try:
                    src = pg.evaluate("b => window.run(b)", b64)
                    (raw_dir / f.name).write_bytes(base64.b64decode(src.split(",", 1)[1]))
                    calls += 1
                    break
                except Exception as e:
                    if calls == 1 and not reload_every:
                        reload_every = 1
                        tqdm.write("2nd transfer on a page fails here; reloading the model before every frame from now on")
                    tqdm.write(f"frame {i} failed ({str(e)[:400]}), reloading model")
                    pg.close(); pg = new_page(); calls = 0
        browser.close()
    srv.shutdown()
    shutil.rmtree(work, ignore_errors=True)


# ---------- encode ----------
def encode(out_dir, out_path, fps):
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
                        "-i", str(out_dir / "%05d.png"), "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", str(out_path)], check=True)
    else:
        print("ffmpeg not found, using OpenCV mp4v (less compatible)")
        w = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (DISPLAY, DISPLAY))
        for f in sorted(out_dir.glob("*.png")):
            w.write(cv2.imread(str(f)))
        w.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to .pict file")
    ap.add_argument("--input", required=True, help="folder of images, a video, a gif, or one image")
    ap.add_argument("--out", default="transferred-video.mp4")
    ap.add_argument("--fps", type=float, default=None, help="output fps (default: 8 for image folders, source fps for video/gif)")
    ap.add_argument("--frame-step", type=int, default=1, help="keep every Nth frame (video/gif)")
    ap.add_argument("--max-frames", type=int, default=1000)
    ap.add_argument("--no-dither", action="store_true")
    ap.add_argument("--headed", action="store_true", help="show the browser window (default on macOS, where headless Chromium is CPU-only)")
    ap.add_argument("--headless", action="store_true", help="force headless (CPU WebGL on macOS)")
    ap.add_argument("--debug", action="store_true", help="print browser-side errors")
    ap.add_argument("--swiftshader", action="store_true", help="force software WebGL (fallback if GPU fails)")
    ap.add_argument("--work-dir", default=None, help="where intermediate frames go (default: <out name>_work next to where you run the script)")
    ap.add_argument("--fresh", action="store_true", help="delete the work dir first instead of resuming")
    ap.add_argument("--keep", action="store_true", help="keep the work dir after a successful run")
    a = ap.parse_args()

    work = Path(a.work_dir or (Path(a.out).stem + "_work")).resolve()
    if a.fresh:
        shutil.rmtree(work, ignore_errors=True)
    frames, raw, out = work / "in", work / "raw", work / "out"
    for d in (frames, out):                       # rebuilt every run
        shutil.rmtree(d, ignore_errors=True)
    for d in (frames, raw, out):
        d.mkdir(parents=True, exist_ok=True)      # raw/ is kept so a crashed run can resume
    done = len(list(raw.glob("*.png")))
    print(f"work dir: {work}" + (f"  (resuming, {done} frames already processed)" if done else ""))

    n, fps = prepare_frames(a.input, frames, a.frame_step, a.max_frames, a.fps)
    if n == 0:
        sys.exit("No frames found in input.")
    print(f"{n} frames prepared, output fps = {fps:g}")

    headed = a.headed or (sys.platform == 'darwin' and not a.headless)
    run_model(a.model, frames, raw, headed, a.swiftshader, a.debug)

    print("dithering / upscaling...")
    for f in tqdm(sorted(raw.glob("*.png")), unit="frame", ncols=100):
        im = Image.open(f).convert("RGB").resize((DISPLAY, DISPLAY), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32).copy()
        if not a.no_dither:
            fs_dither(arr, 4)
        Image.fromarray(arr.astype(np.uint8)).save(out / f.name)

    encode(out, Path(a.out), fps)
    print(f"done -> {Path(a.out).resolve()}")
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()