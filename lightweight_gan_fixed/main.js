let session = null;
let walkingLatent = false;
let stopLatentWalk = false;

const genBtn = document.getElementById('genBtn');
const latentWalkBtn = document.getElementById('latentWalkBtn');
const statusEl = document.getElementById('status');
const canvas = document.getElementById('output');
const placeholderText = document.getElementById('placeholderText');
const latentShapeInput = document.getElementById('latentShape');
const pixelRangeSelect = document.getElementById('pixelRange');

function setStatus(msg, kind) {
  statusEl.textContent = msg;
  statusEl.className = kind || '';
}

async function loadModel() {
  genBtn.disabled = true;
  latentWalkBtn.disabled = true;
  setStatus('Loading model...');

  try {
    session = await ort.InferenceSession.create('./model.onnx', {
      executionProviders: ['wasm']
    });

    setStatus(
      'Model loaded. Input: "' + session.inputNames[0] +
      '", output: "' + session.outputNames[0] + '".',
      'ok'
    );
    genBtn.disabled = false;
    latentWalkBtn.disabled = false;
  } catch (err) {
    console.error(err);
    setStatus('Failed to load model: ' + err.message, 'err');
    session = null;
  }
}

loadModel();

genBtn.addEventListener('click', async () => {
  if (!session) return;
  genBtn.disabled = true;
  setStatus('Generating...');

  try {
    const shape = '1,256'.split(',').map(s => parseInt(s.trim(), 10));
    if (shape.some(isNaN)) throw new Error('Invalid latent shape.');

    const size = shape.reduce((a, b) => a * b, 1);
    const noiseData = new Float32Array(size);

    for (let i = 0; i < size; i++) {
      const u1 = Math.random() || 1e-6;
      const u2 = Math.random();
      noiseData[i] = Math.sqrt(-2 * Math.log(u1)) *
                     Math.cos(2 * Math.PI * u2);
    }

    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];
    const inputTensor = new ort.Tensor('float32', noiseData, shape);

    const results = await session.run({ [inputName]: inputTensor });
    renderTensorToCanvas(results[outputName], 'sigmoid');
    setStatus('Done.', 'ok');
  } catch (err) {
    console.error(err);
    setStatus('Generation failed: ' + err.message, 'err');
  } finally {
    genBtn.disabled = false;
  }
});

function renderTensorToCanvas(tensor, rangeMode) {
  const dims = tensor.dims;
  let C, H, W, layout;

  if (dims.length === 4) {
    if (dims[1] <= 4) {
      C = dims[1];
      H = dims[2];
      W = dims[3];
      layout = 'CHW';
    } else {
      H = dims[1];
      W = dims[2];
      C = dims[3];
      layout = 'HWC';
    }
  } else if (dims.length === 3) {
    C = 1;
    H = dims[1];
    W = dims[2];
    layout = 'CHW';
  } else {
    throw new Error('Unexpected output shape: [' + dims.join(',') + ']');
  }

  const data = tensor.data;
  canvas.width = W;
  canvas.height = H;

  const ctx = canvas.getContext('2d');
  const imgData = ctx.createImageData(W, H);

  const toByte = (v) => {
    let n = rangeMode === 'tanh' ? (v + 1) / 2 : v;
    n = Math.max(0, Math.min(1, n));
    return Math.round(n * 255);
  };

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const pixelIdx = (y * W + x) * 4;
      let r, g, b;

      if (C === 1) {
        const val = data[y * W + x];
        const byte = toByte(val);
        r = g = b = byte;
      } else if (layout === 'CHW') {
        r = toByte(data[0 * H * W + y * W + x]);
        g = toByte(data[1 * H * W + y * W + x]);
        b = toByte(data[2 * H * W + y * W + x]);
      } else {
        const offset = (y * W + x) * C;
        r = toByte(data[offset]);
        g = toByte(data[offset + 1]);
        b = toByte(data[offset + 2]);
      }

      imgData.data[pixelIdx] = r;
      imgData.data[pixelIdx + 1] = g;
      imgData.data[pixelIdx + 2] = b;
      imgData.data[pixelIdx + 3] = 255;
    }
  }

  ctx.putImageData(imgData, 0, 0);
  canvas.style.display = 'block';
  placeholderText.style.display = 'none';
}

async function generateFromLatent(noiseData, shape) {
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];
  const inputTensor = new ort.Tensor('float32', noiseData, shape);
  const results = await session.run({ [inputName]: inputTensor });
  renderTensorToCanvas(results[outputName], 'sigmoid');
}

function randomLatent(size) {
  const noiseData = new Float32Array(size);

  for (let i = 0; i < size; i++) {
    const u1 = Math.random() || 1e-6;
    const u2 = Math.random();
    noiseData[i] = Math.sqrt(-2 * Math.log(u1)) *
                   Math.cos(2 * Math.PI * u2);
  }

  return noiseData;
}

async function runLatentWalk() {
  if (!session) return;

  if (walkingLatent) {
    stopLatentWalk = true;
    return;
  }

  walkingLatent = true;
  stopLatentWalk = false;
  genBtn.disabled = true;
  latentWalkBtn.textContent = 'Stop latent walk';
  setStatus('Walking between two latent points...');

  try {
    const shape = [1, 256];
    const size = 256;
    const frames = 60;

    const first = randomLatent(size);
    const second = randomLatent(size);
    let reverse = false;

    while (!stopLatentWalk) {
      for (let frame = 0; frame <= frames && !stopLatentWalk; frame++) {
        const progress = frame / frames;
        const t = reverse ? 1 - progress : progress;
        const interpolated = new Float32Array(size);

        for (let i = 0; i < size; i++) {
          interpolated[i] = first[i] + (second[i] - first[i]) * t;
        }

        await generateFromLatent(interpolated, shape);
        await new Promise(resolve => setTimeout(resolve, 80));
      }

      reverse = !reverse;
    }

    setStatus('Latent space walk stopped.', 'ok');
  } catch (err) {
    console.error(err);
    setStatus('Latent space walk failed: ' + err.message, 'err');
  } finally {
    walkingLatent = false;
    stopLatentWalk = false;
    genBtn.disabled = false;
    latentWalkBtn.disabled = false;
    latentWalkBtn.textContent = 'Latent space walk';
  }
}

latentWalkBtn.addEventListener('click', runLatentWalk);