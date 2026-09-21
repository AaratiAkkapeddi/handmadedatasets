let session = null;

const modelFile = document.getElementById('modelFile');
const genBtn = document.getElementById('genBtn');
const statusEl = document.getElementById('status');
const canvas = document.getElementById('output');
const placeholderText = document.getElementById('placeholderText');
const latentShapeInput = document.getElementById('latentShape');
const pixelRangeSelect = document.getElementById('pixelRange');

function setStatus(msg, kind){
  statusEl.textContent = msg;
  statusEl.className = kind || '';
}

modelFile.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if(!file) return;
  genBtn.disabled = true;
  setStatus('Loading model...');
  try{
    const buf = await file.arrayBuffer();
    session = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'] });
    setStatus('Model loaded. Input: "' + session.inputNames[0] + '", output: "' + session.outputNames[0] + '".', 'ok');
    genBtn.disabled = false;
  }catch(err){
    console.error(err);
    setStatus('Failed to load model: ' + err.message, 'err');
    session = null;
  }
});

genBtn.addEventListener('click', async () => {
  if(!session) return;
  genBtn.disabled = true;
  setStatus('Generating...');
  try{
    const shape = "1,256".split(',').map(s => parseInt(s.trim(), 10));
    if(shape.some(isNaN)) throw new Error('Invalid latent shape.');

    const size = shape.reduce((a,b) => a*b, 1);
    const noiseData = new Float32Array(size);
    for(let i=0; i<size; i++){
      // standard normal noise via Box-Muller
      const u1 = Math.random() || 1e-6;
      const u2 = Math.random();
      noiseData[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    const inputName = session.inputNames[0];
    const outputName = session.outputNames[0];
    const inputTensor = new ort.Tensor('float32', noiseData, shape);

    const results = await session.run({ [inputName]: inputTensor });
    const outTensor = results[outputName];
    renderTensorToCanvas(outTensor, "sigmoid");

    setStatus('Done.', 'ok');
  }catch(err){
    console.error(err);
    setStatus('Generation failed: ' + err.message, 'err');
  }finally{
    genBtn.disabled = false;
  }
});

function renderTensorToCanvas(tensor, rangeMode){
  const dims = tensor.dims; // expect [1, C, H, W] or [1, H, W, C] or [1, H, W]
  let C, H, W, layout;

  if(dims.length === 4){
    if(dims[1] <= 4){ C = dims[1]; H = dims[2]; W = dims[3]; layout = 'CHW'; }
    else { H = dims[1]; W = dims[2]; C = dims[3]; layout = 'HWC'; }
  }else if(dims.length === 3){
    C = 1; H = dims[1]; W = dims[2]; layout = 'CHW';
  }else{
    throw new Error('Unexpected output shape: [' + dims.join(',') + ']');
  }

  const data = tensor.data;
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');
  const imgData = ctx.createImageData(W, H);

  const toByte = (v) => {
    let n;
    if(rangeMode === 'tanh'){ n = (v + 1) / 2; }
    else { n = v; }
    n = Math.max(0, Math.min(1, n));
    return Math.round(n * 255);
  };

  for(let y=0; y<H; y++){
    for(let x=0; x<W; x++){
      const pixelIdx = (y * W + x) * 4;
      let r, g, b;
      if(C === 1){
        let val;
        if(layout === 'CHW') val = data[y*W + x];
        else val = data[(y*W + x)];
        const byte = toByte(val);
        r = g = b = byte;
      }else{
        if(layout === 'CHW'){
          r = toByte(data[0*H*W + y*W + x]);
          g = toByte(data[1*H*W + y*W + x]);
          b = toByte(data[2*H*W + y*W + x]);
        }else{
          r = toByte(data[(y*W + x)*C + 0]);
          g = toByte(data[(y*W + x)*C + 1]);
          b = toByte(data[(y*W + x)*C + 2]);
        }
      }
      imgData.data[pixelIdx] = r;
      imgData.data[pixelIdx+1] = g;
      imgData.data[pixelIdx+2] = b;
      imgData.data[pixelIdx+3] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);
  canvas.style.display = 'block';
  placeholderText.style.display = 'none';
}