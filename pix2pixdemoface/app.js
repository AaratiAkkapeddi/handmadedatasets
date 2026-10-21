const SIZE = 256;
const SCALE = 2;
const DISPLAY = SIZE * SCALE; // 512
const FACE_WIDTH = 0.43; // brow-to-brow width as a fraction of the canvas (matches your samples)
const EYE_Y = 0.5;       // eye line height as a fraction of the canvas

let inputImg, inputCanvas, modelCanvas, output, statusMsg, pix2pix, outputCanvas;
let predictions = [];
let videoWidth = 0;
let videoHeight = 0;
let randomBtn, clearBtn, transferBtn;
let showingExample = false;
let pix2pixReady = false;
let faceMeshReady = false;
let cameraReady = false;

const FEATURES = [
  { parts: ['leftEyebrowUpper'], closed: false, trimEnd: 2, trimBeginning: 2 },
  { parts: ['rightEyebrowUpper'], closed: false, trimEnd: 2, trimBeginning: 2 },
  { parts: ['leftEyeUpper0', 'leftEyeLower0'], closed: true },
  { parts: ['rightEyeUpper0', 'rightEyeLower0'], closed: true },
  { parts: ['noseLeftCorner', 'noseBottom', 'noseRightCorner'], closed: false },
  { parts: ['lipsUpperInner', 'lipsLowerInner'], closed: true }
];

const INTERVAL = 200; // ms between generations
let busy = false;
let lastRun = 0;

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
  outputCanvas = createGraphics(DISPLAY, DISPLAY);
  outputCanvas.pixelDensity(1);

  background(0);

  output    = select('#output');
  statusMsg = select('#status');
  statusMsg.html('Allow camera access to start face landmark tracking...');

  window.addEventListener('message', function(event) {
    if (event.origin !== window.location.origin) return;
    if (!event.data || event.source !== select('#faceTracker').elt.contentWindow) return;

    if (event.data.type === 'face-tracker-ready') {
      cameraReady = true;
      faceMeshReady = true;
      updateStatus();
    } else if (event.data.type === 'face-tracker-predictions') {
      predictions = event.data.predictions || [];
      videoWidth = event.data.width;
      videoHeight = event.data.height;
    } else if (event.data.type === 'face-tracker-error') {
      statusMsg.html('Face tracking failed: ' + event.data.message);
    }
  });

  // transferBtn = select('#transferBtn');
  clearBtn    = select('#clearBtn');
  clearBtn.mousePressed(clearCanvas);

  randomBtn = select('#randomBtn');
  randomBtn.mousePressed(function() {
    let src = ['images/seg0.png','images/seg1.png','images/seg2.png','images/seg3.png',
               'images/seg4.png','images/seg5.png','images/seg6.png','images/seg7.png',
               'images/seg8.png','images/seg9.png','images/seg10.png','images/seg11.png',
               'images/seg12.png'];
    let index = int(random(0, 13));
    showingExample = true;
    inputImg = loadImage(src[index], () => {
      background(0);
      image(inputImg, 0, 0, DISPLAY, DISPLAY);
    });
  });

  describe(
    'A 512 by 512 canvas showing white facial landmark outlines on a black background, tracked from the camera.'
  );

  pix2pix = ml5.pix2pix('model/faCES.pict', modelLoaded);
}

function draw() {
  if (!showingExample) drawLandmarks();

  if (pix2pixReady && faceMeshReady && cameraReady &&
      !busy && millis() - lastRun >= INTERVAL) {
    lastRun = millis();
    runTransferCycle();
  }
}

async function runTransferCycle() {
  busy = true;
  let stage = 'Face landmark tracking';
  try {
    if (!showingExample) {
      if (predictions.length === 0) {
        statusMsg.html('No face detected. Center your face in the camera.');
        return;
      }
      drawLandmarks();
    }
    stage = 'Pix2Pix transfer';
    await transfer();
  } catch (error) {
    console.error(stage + ' failed:', error);
    statusMsg.html(stage + ' failed: ' + (error.message || error));
  } finally {
    busy = false;
  }
}

function drawLandmarks() {
  background(0);
  stroke(255);
  strokeWeight(1);
  strokeCap(ROUND);
  strokeJoin(ROUND);
  noFill();

  if (predictions.length === 0) return;

  const a = predictions[0].annotations;
  const vw = videoWidth;
  const vh = videoHeight;
  if (!a || vw === 0 || vh === 0) return;

  const leftBrow = (a.leftEyebrowUpper || []).slice(0, -2);
  const rightBrow = (a.rightEyebrowUpper || []).slice(0, -2);
  const isPoint = point => point && Number.isFinite(point[0]) && Number.isFinite(point[1]);
  const browPoints = leftBrow.concat(rightBrow).filter(isPoint);
  const eyePoints = (a.leftEyeUpper0 || []).concat(a.rightEyeUpper0 || []).filter(isPoint);

  if (browPoints.length < 2 || eyePoints.length === 0) return;

  const minX = Math.min(...browPoints.map(point => point[0]));
  const maxX = Math.max(...browPoints.map(point => point[0]));
  const eyeY = eyePoints.reduce((sum, point) => sum + point[1], 0) / eyePoints.length;
  const browWidth = maxX - minX;
  if (browWidth <= 0) return;

  const s = (FACE_WIDTH * DISPLAY) / browWidth;
  const midX = (minX + maxX) / 2;
  const toX = x => DISPLAY / 2 - (x - midX) * s;
  const toY = y => EYE_Y * DISPLAY + (y - eyeY) * s;

  FEATURES.forEach(function(f) {
    const pts = [];
    f.parts.forEach((name, i) => {
      if (!a[name]) return;
      let group = [...a[name]];
      if (f.trimEnd) group = group.slice(0, group.length - f.trimEnd);
      if (i % 2) group.reverse();
      pts.push(...group);
    });
    if (pts.length < 2) return;
    beginShape();
    pts.forEach(p => vertex(toX(p[0]), toY(p[1])));
    if (f.closed) endShape(CLOSE); else endShape();
  });


  // // Mouth: one line between the lip corners
  // const up = a.lipsUpperOuter;
  // if (up && up.length > 1) {
  //   const l = up[0], r = up[up.length - 1];
  //   line(toX(l[0]), toY(l[1]), toX(r[0]), toY(r[1]));
  // }
}

function modelLoaded() {
  pix2pixReady = true;
  updateStatus();
}

function updateStatus() {
  if (!cameraReady) {
    statusMsg.html('Allow camera access to start face landmark tracking...');
  } else if (pix2pixReady && faceMeshReady) {
    statusMsg.html('Face tracking ready. Center your face in the camera.');
  } else if (pix2pixReady) {
    statusMsg.html('Pix2Pix model loaded. Loading face landmark model...');
  } else if (faceMeshReady) {
    statusMsg.html('Face landmark model loaded. Loading Pix2Pix model...');
  }
}

function clearCanvas() {
  showingExample = false;
  background(0);
  output.elt.src = '';
  output.elt.alt = 'The AI-generated face output will appear here after clicking Transfer.';
  updateStatus();
}

async function transfer() {
  modelCanvas.image(get(), 0, 0, SIZE, SIZE);

  const result = await new Promise((resolve, reject) => {
    pix2pix.transfer(modelCanvas.elt, function(error, transferResult) {
      if (error) reject(error);
      else resolve(transferResult);
    });
  });
  if (!result || !result.src) {
    throw new Error('The model returned no image.');
  }

  const generatedImage = await new Promise((resolve, reject) => {
    loadImage(result.src, resolve, reject);
  });
  outputCanvas.clear();
  outputCanvas.image(generatedImage, 0, 0, DISPLAY, DISPLAY);
  ditherFloydSteinbergColor(outputCanvas);

  const blob = await new Promise((resolve, reject) => {
    outputCanvas.canvas.toBlob(resultBlob => {
      if (resultBlob) resolve(resultBlob);
      else reject(new Error('Could not create the generated image.'));
    });
  });
  const old = output.elt.src;
  output.elt.src = URL.createObjectURL(blob);
  if (old.startsWith('blob:')) URL.revokeObjectURL(old);
  output.elt.alt = 'AI-generated face produced from the tracked facial landmarks.';
  statusMsg.html('Generating live, once per second.');
}