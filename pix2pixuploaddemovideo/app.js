const SIZE = 256;
const SCALE = 2;
const DISPLAY = SIZE * SCALE; // 512
const STROKE_WEIGHT = 0.5;

// --- Image-sequence processing settings ---
const OUTPUT_FPS = 8;             // playback rate of the stitched output video
const MAX_SEQUENCE_FRAMES = 1000; // safety cap so an enormous folder can't run forever

let inputImg, inputCanvas, modelCanvas, output, statusMsg;
let pix2pix, transferBtn, clearBtn, modelFileInput, imageFileInput;
let isDrawing = false;
let currentModelUrl = null;

// --- Image-sequence state ---
let uploadedImageFiles = [];      // sorted array of File objects (the frame sequence)
let imageFolderInput, sequenceTransferBtn, sequenceOutput, sequenceDownloadLink;
let processingSequence = false;

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

  setupSequenceControls();
}

// Wires up the image-sequence UI. Picks up existing elements by id if the
// HTML already defines them (#imageFolderInput, #videoTransferBtn,
// #videoOutput -- kept as the ids so existing markup / CSS doesn't need to
// change), otherwise creates them.
function setupSequenceControls() {
  const existingInput = document.getElementById('imageFolderInput');
  const existingBtn    = document.getElementById('videoTransferBtn');
  const existingOutput = document.getElementById('videoOutput');

  const controlsHost = transferBtn.elt.parentNode;

  const wrapper = document.createElement('div');
  wrapper.style.marginTop = '12px';

  if (existingInput) {
    imageFolderInput = existingInput;
  } else {
    imageFolderInput = document.createElement('input');
    imageFolderInput.type = 'file';
    imageFolderInput.id = 'imageFolderInput';
    imageFolderInput.accept = 'image/*';
    imageFolderInput.multiple = true;
    imageFolderInput.setAttribute('webkitdirectory', '');
    imageFolderInput.setAttribute('directory', '');
    wrapper.appendChild(imageFolderInput);
  }
  imageFolderInput.addEventListener('change', handleImageFolder);

  if (existingBtn) {
    sequenceTransferBtn = existingBtn;
  } else {
    sequenceTransferBtn = document.createElement('button');
    sequenceTransferBtn.id = 'videoTransferBtn';
    sequenceTransferBtn.textContent = 'Transfer Image Sequence';
    wrapper.appendChild(sequenceTransferBtn);
  }
  // Note: intentionally NOT using the `disabled` attribute here. A truly
  // disabled button blocks clicks entirely, so the user gets no feedback
  // about *why* nothing happens. Instead we keep it clickable and let
  // transferImageSequence() explain what's missing (no model / no images
  // yet), while updateSequenceButtonState() below handles the visual dimming.
  sequenceTransferBtn.removeAttribute('disabled');
  sequenceTransferBtn.addEventListener('click', transferImageSequence);
  updateSequenceButtonState();

  if (!existingInput && !existingBtn) {
    controlsHost.appendChild(wrapper);
  }

  if (existingOutput) {
    sequenceOutput = existingOutput;
  } else {
    sequenceOutput = document.createElement('video');
    sequenceOutput.id = 'videoOutput';
    sequenceOutput.controls = true;
    sequenceOutput.style.maxWidth = `${DISPLAY}px`;
    sequenceOutput.style.display = 'none';
    output.elt.parentNode.appendChild(sequenceOutput);
  }

  sequenceDownloadLink = document.createElement('a');
  sequenceDownloadLink.textContent = 'Download video';
  sequenceDownloadLink.style.display = 'none';
  sequenceDownloadLink.style.marginLeft = '8px';
  sequenceOutput.parentNode.appendChild(sequenceDownloadLink);
}

// Visually dims the sequence button when a model or image sequence hasn't
// been loaded yet, WITHOUT using the disabled attribute -- so clicks still
// register and transferImageSequence()'s own checks can tell the user
// exactly what's missing.
function updateSequenceButtonState() {
  if (!sequenceTransferBtn) return;
  const ready = !!pix2pix && uploadedImageFiles.length > 0;
  sequenceTransferBtn.classList.toggle('is-disabled', !ready);
  sequenceTransferBtn.setAttribute('aria-disabled', String(!ready));
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
  updateSequenceButtonState();
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
  updateSequenceButtonState();
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

// ── Image sequence: upload ──────────────────────────────────────────────
// Natural sort so "frame2.png" sorts before "frame10.png" instead of after.
function naturalCompare(a, b) {
  const ax = [], bx = [];
  a.replace(/(\d+)|(\D+)/g, (_, d, s) => ax.push([d ? Number(d) : Infinity, s || '']));
  b.replace(/(\d+)|(\D+)/g, (_, d, s) => bx.push([d ? Number(d) : Infinity, s || '']));
  while (ax.length && bx.length) {
    const an = ax.shift(), bn = bx.shift();
    const nn = (an[0] - bn[0]) || an[1].localeCompare(bn[1]);
    if (nn) return nn;
  }
  return ax.length - bx.length;
}

function handleImageFolder(evt) {
  const files = Array.from(evt.target.files || []).filter(f => f.type.startsWith('image/'));

  uploadedImageFiles = files.sort((a, b) => {
    const an = a.webkitRelativePath || a.name;
    const bn = b.webkitRelativePath || b.name;
    return naturalCompare(an, bn);
  });

  updateSequenceButtonState();

  if (uploadedImageFiles.length === 0) {
    statusMsg.html('No image files found in that selection.');
    return;
  }

  if (uploadedImageFiles.length > MAX_SEQUENCE_FRAMES) {
    statusMsg.html(
      `Found ${uploadedImageFiles.length} images, but only the first ${MAX_SEQUENCE_FRAMES} will be processed.`
    );
    uploadedImageFiles = uploadedImageFiles.slice(0, MAX_SEQUENCE_FRAMES);
  } else {
    statusMsg.html(
      `Loaded ${uploadedImageFiles.length} images (will render at ${OUTPUT_FPS}fps). ` +
      (pix2pix ? 'Click "Transfer Image Sequence" to begin.' : 'Upload a .pict model file, then click "Transfer Image Sequence".')
    );
  }
}

// ── Image sequence: helpers ──────────────────────────────────────────────
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

// Discards the current pix2pix instance and creates a fresh one from the
// same .pict file. Used to recover mid-run if the model's WebGL context gets
// evicted (GPU memory pressure) -- a brand-new instance gets brand-new
// textures for its weights instead of trying to repair the old, corrupted
// ones.
function reloadModel() {
  return new Promise((resolve, reject) => {
    if (!currentModelUrl) {
      reject(new Error('No model file to reload from.'));
      return;
    }
    try {
      pix2pix = ml5.pix2pix(currentModelUrl, () => resolve());
    } catch (err) {
      reject(err);
    }
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

      sequenceOutput.src = url;
      sequenceOutput.style.display = 'block';

      sequenceDownloadLink.href = url;
      sequenceDownloadLink.download = 'transferred-video.webm';
      sequenceDownloadLink.style.display = 'inline';

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

// ── Image sequence: per-frame transfer ───────────────────────────────────
async function transferImageSequence() {
  if (!pix2pix) {
    statusMsg.html('Please upload a .pict model file first.');
    return;
  }
  if (uploadedImageFiles.length === 0) {
    statusMsg.html('Please upload a folder of images first.');
    return;
  }
  if (processingSequence) return;

  processingSequence = true;
  sequenceTransferBtn.classList.add('is-disabled');
  sequenceOutput.style.display = 'none';
  sequenceDownloadLink.style.display = 'none';

  const files = uploadedImageFiles;
  const fps = OUTPUT_FPS;
  const totalFrames = files.length;

  // Two scratch canvases, allocated ONCE and reused every frame, so memory
  // stays flat regardless of how many images are in the sequence. Streaming
  // each processed frame straight into the recorder (rather than buffering
  // every result in an array for stitching at the end) avoids the GPU
  // memory pressure that used to evict the pix2pix model's WebGL context on
  // long runs -- the "Cannot read properties of undefined (reading
  // '...kernel')" errors.
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
    processingSequence = false;
    updateSequenceButtonState();
    return;
  }

  const frameDurationMs = 1000 / fps;
  let framesWritten = 0;
  let consecutiveFailures = 0;
  let reloadAttempts = 0;
  const MAX_CONSECUTIVE_FAILURES = 5; // this many identical failures in a row means the model's context is gone, not a one-off glitch
  const MAX_RELOAD_ATTEMPTS = 3;      // cap total reloads so a persistently broken setup still gives up eventually

  for (let i = 0; i < totalFrames; i++) {
    statusMsg.html(`Loading frame ${i + 1} / ${totalFrames}...`);

    const objectUrl = URL.createObjectURL(files[i]);
    let sourceImg;
    try {
      sourceImg = await loadImageAsync(objectUrl);
    } catch (err) {
      console.log('Frame', i, 'failed to load:', err);
      URL.revokeObjectURL(objectUrl);
      continue;
    }

    // Draw the source image at the model's native resolution.
    frameCanvas.image(sourceImg, 0, 0, SIZE, SIZE);
    URL.revokeObjectURL(objectUrl);

    statusMsg.html(`Processing frame ${i + 1} / ${totalFrames}...`);

    let result;
    try {
      result = await pix2pixTransferAsync(frameCanvas.elt);
      consecutiveFailures = 0;
    } catch (err) {
      console.log('Frame', i, 'failed:', err);
      consecutiveFailures++;

      if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
        if (reloadAttempts >= MAX_RELOAD_ATTEMPTS) {
          statusMsg.html('The model keeps losing its GPU context and reloading isn\u2019t helping. Try a shorter sequence, a lower resolution, or a different device/browser.');
          break;
        }

        reloadAttempts++;
        statusMsg.html(`The model appears to have lost its GPU context (likely memory pressure). Reloading it (attempt ${reloadAttempts}/${MAX_RELOAD_ATTEMPTS})...`);
        try {
          await reloadModel();
          consecutiveFailures = 0;
          i--; // retry this same frame now that the model is fresh
          continue;
        } catch (reloadErr) {
          console.log('Model reload failed:', reloadErr);
          statusMsg.html('Could not reload the model automatically. Try a shorter sequence, or reload the page and try again.');
          break;
        }
      }
      continue;
    }
    if (!result || !result.src) continue;

    let p5img;
    try {
      p5img = await loadImageAsync(result.src);
    } catch (err) {
      console.log('Frame', i, 'image load failed:', err);
      continue;
    } finally {
      // If ml5 handed back a blob URL rather than a data URL, release it --
      // otherwise these accumulate for the life of the page.
      if (result.src.startsWith('blob:')) {
        URL.revokeObjectURL(result.src);
      }
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

  processingSequence = false;
  updateSequenceButtonState();
}