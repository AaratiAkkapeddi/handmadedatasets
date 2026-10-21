const SIZE = 256;
const SCALE = 2;
const DISPLAY = SIZE * SCALE; // 512
const HAND_FRAME_MARGIN = 0.08;

let inputImg, inputCanvas, modelCanvas, output, statusMsg, pix2pix, outputCanvas;
let predictions = [];
let videoWidth = 0;
let videoHeight = 0;
let randomBtn, clearBtn, transferBtn;
let showingExample = false;
let pix2pixReady = false;
let handposeReady = false;
let cameraReady = false;

const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17]
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
  statusMsg.html('Allow camera access to start hand tracking...');

  window.addEventListener('message', function(event) {
    if (event.origin !== window.location.origin) return;
    if (!event.data || event.source !== select('#handTracker').elt.contentWindow) return;

    if (event.data.type === 'hand-tracker-ready') {
      cameraReady = true;
      handposeReady = true;
      updateStatus();
    } else if (event.data.type === 'hand-tracker-predictions') {
      predictions = event.data.predictions || [];
      videoWidth = event.data.width;
      videoHeight = event.data.height;
    } else if (event.data.type === 'hand-tracker-error') {
      statusMsg.html('Hand tracking failed: ' + event.data.message);
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
    'A 512 by 512 canvas showing a white hand landmark skeleton on a black background, tracked from the camera.'
  );

  pix2pix = ml5.pix2pix('model/meredith_hand.pict', modelLoaded);
}

function draw() {
  if (!showingExample) drawLandmarks();

  if (pix2pixReady && handposeReady && cameraReady &&
      !busy && millis() - lastRun >= INTERVAL) {
    lastRun = millis();
    runTransferCycle();
  }
}

async function runTransferCycle() {
  busy = true;
  let stage = 'Hand landmark tracking';
  try {
    if (!showingExample) {
      if (predictions.length === 0) {
        statusMsg.html('No hand detected. Show your hand to the camera.');
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
  strokeWeight(3);
  strokeCap(ROUND);
  strokeJoin(ROUND);
  noFill();

  if (predictions.length === 0) return;

  const landmarks = predictions[0].landmarks;
  const vw = videoWidth;
  const vh = videoHeight;
  const isPoint = point => point && Number.isFinite(point[0]) && Number.isFinite(point[1]);
  if (!landmarks || landmarks.length !== 21 || vw === 0 || vh === 0 ||
      !landmarks.every(isPoint)) return;

  const minX = Math.min(...landmarks.map(point => point[0]));
  const maxX = Math.max(...landmarks.map(point => point[0]));
  const minY = Math.min(...landmarks.map(point => point[1]));
  const maxY = Math.max(...landmarks.map(point => point[1]));
  const handWidth = maxX - minX;
  const handHeight = maxY - minY;
  if (handWidth <= 0 || handHeight <= 0) return;

  const availableSize = DISPLAY * (1 - HAND_FRAME_MARGIN * 2);
  const scale = Math.min(availableSize / handWidth, availableSize / handHeight);
  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const toX = x => DISPLAY / 2 - (x - centerX) * scale;
  const toY = y => DISPLAY / 2 + (y - centerY) * scale;

  HAND_CONNECTIONS.forEach(([start, end]) => {
    const from = landmarks[start];
    const to = landmarks[end];
    line(toX(from[0]), toY(from[1]), toX(to[0]), toY(to[1]));
  });
}

function modelLoaded() {
  pix2pixReady = true;
  updateStatus();
}

function updateStatus() {
  if (!cameraReady) {
    statusMsg.html('Allow camera access to start hand tracking...');
  } else if (pix2pixReady && handposeReady) {
    statusMsg.html('Hand tracking ready. Show your hand to the camera.');
  } else if (pix2pixReady) {
    statusMsg.html('Pix2Pix model loaded. Loading hand landmark model...');
  } else if (handposeReady) {
    statusMsg.html('Hand landmark model loaded. Loading Pix2Pix model...');
  }
}

function clearCanvas() {
  showingExample = false;
  background(0);
  output.elt.src = '';
  output.elt.alt = 'The AI-generated hand output will appear here after clicking Transfer.';
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
  output.elt.alt = 'AI-generated hand produced from the tracked hand landmarks.';
  statusMsg.html('Generating live, once per second.');
}