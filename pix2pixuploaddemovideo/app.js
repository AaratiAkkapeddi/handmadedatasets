const SIZE = 256;
const SCALE = 2;
const DISPLAY = SIZE * SCALE; // 512
const STROKE_WEIGHT = 0.5;

// --- Video processing settings ---
const VIDEO_SAMPLE_FPS = 8;      // how many frames/sec to sample from the source video
const MAX_VIDEO_FRAMES = 240;    // safety cap (30s @ 8fps) so a long upload can't run forever

let inputImg, inputCanvas, modelCanvas, output, statusMsg;
let pix2pix, transferBtn, clearBtn, modelFileInput, imageFileInput;
let isDrawing = false;
let currentModelUrl = null;

// --- Video state ---
let uploadedVideo = null;        // hidden <video> element holding the uploaded clip
let uploadedVideoUrl = null;     // object URL for the uploaded clip
let videoFileInput, videoTransferBtn, videoOutput, videoDownloadLink;
let processingVideo = false;

// ── Floyd-Steinberg dither (color) ───────────────────────────────────────────
function ditherFloydSteinbergColor(pg) {
  pg.loadPixels();
  const w = pg.width, h = pg.height;

  const r = new Float32Array(w * h);
  const g = new Float32Array(w * h);
  const b = new Float32Array(w * h);

  for (let i = 0; i < w * h; i++) {
    const o = i * 4;
    r[i] = pg.pixels[o];
    g[i] = pg.pixels[o + 1];
    b[i] = pg.pixels[o + 2];
  }

  const levels = 4;
  const step = 255 / (levels - 1);
  const clamp = v => Math.min(255, Math.max(0, v));
  const quantize = v => Math.round(Math.round(v / step) * step);

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const idx = y * w + x;
      const oldR = r[idx], oldG = g[idx], oldB = b[idx];
      const newR = quantize(oldR), newG = quantize(oldG), newB = quantize(oldB);
      r[idx] = newR; g[idx] = newG; b[idx] = newB;
      const errR = oldR - newR, errG = oldG - newG, errB = oldB - newB;
      if (x + 1 < w) {
        r[idx+1] = clamp(r[idx+1] + errR*7/16);
        g[idx+1] = clamp(g[idx+1] + errG*7/16);
        b[idx+1] = clamp(b[idx+1] + errB*7/16);
      }
      if (y + 1 < h) {
        if (x - 1 >= 0) {
          r[idx+w-1] = clamp(r[idx+w-1] + errR*3/16);
          g[idx+w-1] = clamp(g[idx+w-1] + errG*3/16);
          b[idx+w-1] = clamp(b[idx+w-1] + errB*3/16);
        }
        r[idx+w] = clamp(r[idx+w] + errR*5/16);
        g[idx+w] = clamp(g[idx+w] + errG*5/16);
        b[idx+w] = clamp(b[idx+w] + errB*5/16);
        if (x + 1 < w) {
          r[idx+w+1] = clamp(r[idx+w+1] + errR*1/16);
          g[idx+w+1] = clamp(g[idx+w+1] + errG*1/16);
          b[idx+w+1] = clamp(b[idx+w+1] + errB*1/16);
        }
      }
    }
  }

  for (let i = 0; i < w * h; i++) {
    const o = i * 4;
    pg.pixels[o]   = r[i];
    pg.pixels[o+1] = g[i];
    pg.pixels[o+2] = b[i];
    pg.pixels[o+3] = 255;
  }
  pg.updatePixels();
}

function setup() {
  pixelDensity(1);

  inputCanvas = createCanvas(DISPLAY, DISPLAY);
  inputCanvas.class('border-box').parent('input');

  modelCanvas = createGraphics(SIZE, SIZE);
  modelCanvas.pixelDensity(1);

  background(0);

  output    = select('#output');
  statusMsg = select('#status');

  transferBtn = select('#transferBtn');
  clearBtn    = select('#clearBtn');
  clearBtn.mousePressed(clearCanvas);

  modelFileInput = select('#modelFileInput');
  modelFileInput.elt.addEventListener('change', handleModelFile);

  imageFileInput = select('#imageFileInput');
  imageFileInput.elt.addEventListener('change', handleImageFile);

  stroke(255);
  strokeWeight(STROKE_WEIGHT);
  strokeCap(ROUND);
  strokeJoin(ROUND);

  // p5 accessibility: describe the canvas for screen readers
  describe(
    'A black 512 by 512 pixel drawing canvas. Use a mouse or touch to sketch a face using thin white lines on a black background, or upload your own image. Draw simple outlines for eyebrows, eyes, a nose, and a mouth, then click the Transfer button to generate an AI-rendered image from your sketch.'
  );

  transferBtn.mousePressed(transfer);

  // No model is loaded until the user uploads a .pict file.
  transferBtn.attribute('disabled', '');

  setupVideoControls();
}

// Builds the video-upload UI programmatically so no HTML edits are required.
// If you'd rather define these in your HTML, just give them the same ids
// (#videoFileInput, #videoTransferBtn, #videoOutput) and this function will
// pick up the existing elements instead of creating new ones.
function setupVideoControls() {
  const existingInput = document.getElementById('videoFileInput');
  const existingBtn    = document.getElementById('videoTransferBtn');
  const existingOutput = document.getElementById('videoOutput');

  const controlsHost = transferBtn.elt.parentNode;

  const wrapper = document.createElement('div');
  wrapper.style.marginTop = '12px';

  if (existingInput) {
    videoFileInput = existingInput;
  } else {
    videoFileInput = document.createElement('input');
    videoFileInput.type = 'file';
    videoFileInput.id = 'videoFileInput';
    videoFileInput.accept = 'video/*';
    wrapper.appendChild(videoFileInput);
  }
  videoFileInput.addEventListener('change', handleVideoFile);

  if (existingBtn) {
    videoTransferBtn = existingBtn;
  } else {
    videoTransferBtn = document.createElement('button');
    videoTransferBtn.id = 'videoTransferBtn';
    videoTransferBtn.textContent = 'Transfer Video';
    wrapper.appendChild(videoTransferBtn);
  }
  // Note: intentionally NOT using the `disabled` attribute here. A truly
  // disabled button blocks clicks entirely, so the user gets no feedback
  // about *why* nothing happens. Instead we keep it clickable and let
  // transferVideo() explain what's missing (no model / no video yet),
  // while updateVideoButtonState() below handles the visual dimming.
  videoTransferBtn.removeAttribute('disabled');
  videoTransferBtn.addEventListener('click', transferVideo);
  updateVideoButtonState();

  if (!existingInput && !existingBtn) {
    controlsHost.appendChild(wrapper);
  }

  if (existingOutput) {
    videoOutput = existingOutput;
  } else {
    videoOutput = document.createElement('video');
    videoOutput.id = 'videoOutput';
    videoOutput.controls = true;
    videoOutput.style.maxWidth = `${DISPLAY}px`;
    videoOutput.style.display = 'none';
    output.elt.parentNode.appendChild(videoOutput);
  }

  videoDownloadLink = document.createElement('a');
  videoDownloadLink.textContent = 'Download video';
  videoDownloadLink.style.display = 'none';
  videoDownloadLink.style.marginLeft = '8px';
  videoOutput.parentNode.appendChild(videoDownloadLink);
}

// Visually dims the video button when a model or video hasn't been loaded
// yet, WITHOUT using the disabled attribute — so clicks still register and
// transferVideo()'s own checks can tell the user exactly what's missing.
function updateVideoButtonState() {
  if (!videoTransferBtn) return;
  const ready = !!pix2pix && !!uploadedVideo;
  videoTransferBtn.classList.toggle('is-disabled', !ready);
  videoTransferBtn.setAttribute('aria-disabled', String(!ready));
}

function draw() {
  if (mouseIsPressed) {
    isDrawing = true;
    stroke(255);
    strokeWeight(STROKE_WEIGHT);
    noFill();
    line(mouseX, mouseY, pmouseX, pmouseY);
  } else {
    isDrawing = false;
  }

  // Update canvas description dynamically based on drawing state
  describeElement(
    inputCanvas.elt,
    isDrawing
      ? 'Drawing in progress on the sketch canvas.'
      : 'Sketch canvas. Draw white lines, or upload your own image, then click Transfer.',
    LABEL
  );
}

function handleModelFile(evt) {
  const file = evt.target.files && evt.target.files[0];
  if (!file) return;

  // Clean up any previously created object URL.
  if (currentModelUrl) {
    URL.revokeObjectURL(currentModelUrl);
    currentModelUrl = null;
  }

  transferBtn.attribute('disabled', '');
  pix2pix = null; // old model is no longer valid while the new one loads
  updateVideoButtonState();
  statusMsg.html('Loading model... Please wait...');

  currentModelUrl = URL.createObjectURL(file);
  pix2pix = ml5.pix2pix(currentModelUrl, modelLoaded);
}

function handleImageFile(evt) {
  const file = evt.target.files && evt.target.files[0];
  if (!file) return;

  const url = URL.createObjectURL(file);
  loadImage(url, img => {
    background(0);
    image(img, 0, 0, DISPLAY, DISPLAY);
    inputImg = img;
    URL.revokeObjectURL(url);
  }, err => {
    console.log(err);
    statusMsg.html('Could not load that image file.');
    URL.revokeObjectURL(url);
  });
}

function modelLoaded() {
  statusMsg.html('Model Loaded!');
  transferBtn.elt.removeAttribute('disabled');
  updateVideoButtonState();
}

function clearCanvas() {
  statusMsg.html(pix2pix ? 'Model Loaded!' : 'Upload a .pict model file to begin.');
  background(0);
  output.elt.src = '';
  output.elt.alt = 'The AI-generated output will appear here after clicking Transfer.';
}

function transfer() {
  if (!pix2pix) {
    statusMsg.html('Please upload a .pict model file first.');
    return;
  }

  statusMsg.html('Transferring...');
  output.elt.alt = 'Generating AI image from your sketch, please wait.';

  modelCanvas.image(get(), 0, 0, SIZE, SIZE);

  pix2pix.transfer(modelCanvas.elt, function(err, result) {
    if (err) { console.log(err); return; }
    if (result && result.src) {
      statusMsg.html('generation done!');

      loadImage(result.src, p5img => {
        const tmp = createGraphics(DISPLAY, DISPLAY);
        tmp.pixelDensity(1);
        tmp.image(p5img, 0, 0, DISPLAY, DISPLAY);
        ditherFloydSteinbergColor(tmp);

        tmp.canvas.toBlob(blob => {
          output.elt.src = URL.createObjectURL(blob);
          output.elt.alt = 'AI-generated image produced from your line drawing. A color-dithered image based on the sketch you drew.';
        });
      });
    }
  });
}

// ── Video: upload ────────────────────────────────────────────────────────
function handleVideoFile(evt) {
  const file = evt.target.files && evt.target.files[0];
  if (!file) return;

  if (uploadedVideoUrl) {
    URL.revokeObjectURL(uploadedVideoUrl);
    uploadedVideoUrl = null;
  }
  uploadedVideo = null;
  updateVideoButtonState();

  const url = URL.createObjectURL(file);
  uploadedVideoUrl = url;

  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.preload = 'auto';
  video.style.display = 'none';
  document.body.appendChild(video); // some browsers need it in the DOM to decode frames reliably

  video.addEventListener('loadedmetadata', () => {
    uploadedVideo = video;
    updateVideoButtonState();
    const frameCount = Math.min(MAX_VIDEO_FRAMES, Math.floor(video.duration * VIDEO_SAMPLE_FPS));
    statusMsg.html(
      `Video loaded (${video.duration.toFixed(1)}s, will process ~${frameCount} frames at ${VIDEO_SAMPLE_FPS}fps). ` +
      (pix2pix ? 'Click "Transfer Video" to begin.' : 'Upload a .pict model file, then click "Transfer Video".')
    );
  });

  video.addEventListener('error', () => {
    statusMsg.html('Could not load that video file.');
  });

  video.src = url;
  video.load();
}

// ── Video: helpers ──────────────────────────────────────────────────────
function seekVideoTo(video, time) {
  return new Promise(resolve => {
    function onSeeked() {
      video.removeEventListener('seeked', onSeeked);
      resolve();
    }
    video.addEventListener('seeked', onSeeked);
    video.currentTime = time;
  });
}

function pix2pixTransferAsync(canvasEl) {
  return new Promise((resolve, reject) => {
    pix2pix.transfer(canvasEl, (err, result) => {
      if (err) reject(err);
      else resolve(result);
    });
  });
}

function loadImageAsync(src) {
  return new Promise((resolve, reject) => {
    loadImage(src, img => resolve(img), err => reject(err));
  });
}

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// Sets up a MediaRecorder against a live canvas and returns a handle to stop
// it. The canvas is drawn into directly by the caller, frame by frame, so
// there's no need to buffer every processed frame in memory.
function startRecording(canvasEl, fps) {
  if (!canvasEl.captureStream) {
    throw new Error('canvas.captureStream is not supported in this browser.');
  }

  const stream = canvasEl.captureStream(fps);
  const mimeCandidates = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
  const mimeType = mimeCandidates.find(m => MediaRecorder.isTypeSupported(m)) || 'video/webm';

  const recorder = new MediaRecorder(stream, { mimeType });
  const chunks = [];
  recorder.ondataavailable = e => {
    if (e.data && e.data.size > 0) chunks.push(e.data);
  };

  const stopped = new Promise((resolve, reject) => {
    recorder.onstop = () => {
      const blob = new Blob(chunks, { type: mimeType });
      const url = URL.createObjectURL(blob);

      videoOutput.src = url;
      videoOutput.style.display = 'block';

      videoDownloadLink.href = url;
      videoDownloadLink.download = 'transferred-video.webm';
      videoDownloadLink.style.display = 'inline';

      resolve();
    };
    recorder.onerror = e => reject(e.error || e);
  });

  recorder.start();

  return {
    stop() {
      recorder.stop();
      return stopped;
    }
  };
}

// ── Video: per-frame transfer ────────────────────────────────────────────
async function transferVideo() {
  if (!pix2pix) {
    statusMsg.html('Please upload a .pict model file first.');
    return;
  }
  if (!uploadedVideo) {
    statusMsg.html('Please upload a video file first.');
    return;
  }
  if (processingVideo) return;

  processingVideo = true;
  videoTransferBtn.classList.add('is-disabled');
  videoOutput.style.display = 'none';
  videoDownloadLink.style.display = 'none';

  const video = uploadedVideo;
  const fps = VIDEO_SAMPLE_FPS;
  const duration = video.duration;
  const totalFrames = Math.min(MAX_VIDEO_FRAMES, Math.max(1, Math.floor(duration * fps)));

  // Two scratch canvases, allocated ONCE and reused every frame. The
  // previous version created a brand-new full-size (512x512) canvas for
  // every processed frame and kept them all in an array until the end for
  // stitching. On longer videos (200+ frames) that built up enough memory
  // pressure that the browser reclaimed GPU memory and evicted the WebGL
  // context the pix2pix model's weights live in -- which is what caused the
  // "Cannot read properties of undefined (reading '...kernel')" errors.
  // Streaming each frame straight into the recording avoids that entirely.
  const frameCanvas = createGraphics(SIZE, SIZE);
  frameCanvas.pixelDensity(1);
  const stitchGfx = createGraphics(DISPLAY, DISPLAY);
  stitchGfx.pixelDensity(1);

  let recorder;
  try {
    recorder = startRecording(stitchGfx.canvas, fps);
  } catch (err) {
    console.log(err);
    statusMsg.html('Could not start recording (see console for details).');
    processingVideo = false;
    updateVideoButtonState();
    return;
  }

  const frameDurationMs = 1000 / fps;
  let framesWritten = 0;

  for (let i = 0; i < totalFrames; i++) {
    const t = Math.min(i / fps, Math.max(0, duration - 0.02));
    await seekVideoTo(video, t);

    // Draw the current video frame at the model's native resolution.
    // NOTE: p5's Graphics.image() only unwraps p5.Image/p5.Element objects
    // (it looks for .canvas or .elt), so it can't handle a raw
    // HTMLVideoElement like the one created in handleVideoFile(). Drawing
    // straight to the native 2D context sidesteps that.
    frameCanvas.drawingContext.drawImage(video, 0, 0, SIZE, SIZE);

    statusMsg.html(`Processing frame ${i + 1} / ${totalFrames}...`);

    let result;
    try {
      result = await pix2pixTransferAsync(frameCanvas.elt);
    } catch (err) {
      console.log('Frame', i, 'failed:', err);
      continue;
    }
    if (!result || !result.src) continue;

    let p5img;
    try {
      p5img = await loadImageAsync(result.src);
    } catch (err) {
      console.log('Frame', i, 'image load failed:', err);
      continue;
    }

    stitchGfx.clear();
    stitchGfx.image(p5img, 0, 0, DISPLAY, DISPLAY);
    ditherFloydSteinbergColor(stitchGfx);

    framesWritten++;
    // Hold this frame on the recorded canvas for one frame-interval so the
    // recorder actually captures it before we move on.
    await delay(frameDurationMs);
  }

  statusMsg.html('Finishing up video...');
  try {
    await recorder.stop();
    statusMsg.html(framesWritten > 0 ? 'Video generation done!' : 'No frames could be processed.');
  } catch (err) {
    console.log(err);
    statusMsg.html('Could not finish recording the video (see console for details).');
  }

  processingVideo = false;
  updateVideoButtonState();
}