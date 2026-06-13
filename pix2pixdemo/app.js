const SIZE = 256;
const SCALE = 2;
const DISPLAY = SIZE * SCALE; // 512
const STROKE_WEIGHT = 0.5;

let inputImg, inputCanvas, modelCanvas, output, statusMsg, pix2pix, randomBtn, clearBtn, transferBtn;
let isDrawing = false;

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

  randomBtn = select('#randomBtn');
  randomBtn.mousePressed(function() {
    let src = ['images/seg0.png','images/seg1.png','images/seg2.png','images/seg3.png',
               'images/seg4.png','images/seg5.png','images/seg6.png','images/seg7.png',
               'images/seg8.png','images/seg9.png','images/seg10.png','images/seg11.png',
               'images/seg12.png'];
    let index = int(random(0, 13));
    inputImg = loadImage(src[index], () => {
      background(0);
      image(inputImg, 0, 0, DISPLAY, DISPLAY);
    });
  });

  stroke(255);
  strokeWeight(STROKE_WEIGHT);
  strokeCap(ROUND);
  strokeJoin(ROUND);

  // p5 accessibility: describe the canvas for screen readers
  describe(
    'A black 512 by 512 pixel drawing canvas. Use a mouse or touch to sketch a face using thin white lines on a black background. Draw simple outlines for eyebrows, eyes, a nose, and a mouth, then click the Transfer button to generate an AI-rendered face from your sketch.'
  );

  pix2pix = ml5.pix2pix('model/humanseg_BtoA.pict', modelLoaded);
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
      ? 'Drawing in progress on the face sketch canvas.'
      : 'Face sketch canvas. Draw white lines to outline eyebrows, eyes, a nose, and a mouth.',
    LABEL
  );
}

function modelLoaded() {
  statusMsg.html('Model Loaded!');
  transferBtn.mousePressed(transfer);
}

function clearCanvas() {
  statusMsg.html('');
  background(0);
  output.elt.src = '';
  output.elt.alt = 'The AI-generated face output will appear here after clicking Transfer.';
}

function transfer() {
  statusMsg.html('Transferring...');
  output.elt.alt = 'Generating AI face from your sketch, please wait.';

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
          output.elt.alt = 'AI-generated face produced from your line drawing. A color-dithered portrait based on the face sketch you drew.';
        });
      });
    }
  });
}