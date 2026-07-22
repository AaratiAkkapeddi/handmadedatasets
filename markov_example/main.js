
(function(){
  const tabPaste = document.getElementById('tab-paste');
  const tabUpload = document.getElementById('tab-upload');
  const panePaste = document.getElementById('pane-paste');
  const paneUpload = document.getElementById('pane-upload');
  const sourceText = document.getElementById('source-text');
  const dropzone = document.getElementById('dropzone');
  const fileInput = document.getElementById('file-input');
  const dzLabel = document.getElementById('dz-label');
  const dzFilename = document.getElementById('dz-filename');
  const wordCount = document.getElementById('word-count');
  const generateBtn = document.getElementById('generate-btn');
  const errorBox = document.getElementById('error-box');
  const outputBody = document.getElementById('output-body');
  const outputMeta = document.getElementById('output-meta');
  const threadPath = document.getElementById('thread-path');
  const ngramSel = document.getElementById('ngram');
  const sentencesSel = document.getElementById('sentences');

  let activeMode = 'paste';
  let uploadedText = '';

  function currentText(){
    return activeMode === 'paste' ? sourceText.value : uploadedText;
  }

  function refreshState(){
    const text = currentText().trim();
    const words = text.length ? text.split(/\s+/).filter(Boolean) : [];
    wordCount.textContent = words.length + ' word' + (words.length === 1 ? '' : 's');
    generateBtn.disabled = words.length < 25;
    errorBox.innerHTML = '';
  }

  tabPaste.addEventListener('click', () => {
    activeMode = 'paste';
    tabPaste.classList.add('active');
    tabUpload.classList.remove('active');
    panePaste.style.display = '';
    paneUpload.style.display = 'none';
    refreshState();
  });

  tabUpload.addEventListener('click', () => {
    activeMode = 'upload';
    tabUpload.classList.add('active');
    tabPaste.classList.remove('active');
    panePaste.style.display = 'none';
    paneUpload.style.display = 'flex';
    refreshState();
  });

  sourceText.addEventListener('input', refreshState);

  function handleFile(file){
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.txt') && file.type !== 'text/plain'){
      errorBox.innerHTML = '<div class="error-msg">That doesn\'t look like a .txt file. Try a plain text file instead.</div>';
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      uploadedText = String(e.target.result || '');
      dzFilename.textContent = file.name + ' · ' + uploadedText.split(/\s+/).filter(Boolean).length + ' words';
      dzLabel.textContent = 'File loaded';
      refreshState();
    };
    reader.onerror = () => {
      errorBox.innerHTML = '<div class="error-msg">Couldn\'t read that file. Please try again.</div>';
    };
    reader.readAsText(file);
  }

  fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

  ['dragenter','dragover'].forEach(evt => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.add('drag');
    });
  });
  ['dragleave','drop'].forEach(evt => {
    dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      dropzone.classList.remove('drag');
    });
  });
  dropzone.addEventListener('drop', (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    handleFile(file);
  });

  function renderWords(str){
    const words = str.split(' ');
    outputBody.innerHTML = '<p class="output-text" id="output-text"></p>';
    const el = document.getElementById('output-text');
    words.forEach((w, i) => {
      const span = document.createElement('span');
      span.className = 'w';
      span.style.animationDelay = (i * 28) + 'ms';
      span.textContent = w + (i < words.length - 1 ? ' ' : '');
      el.appendChild(span);
    });
  }



  generateBtn.addEventListener('click', () => {
    const text = currentText().trim();
    errorBox.innerHTML = '';
    if (typeof RiTa === 'undefined'){
      errorBox.innerHTML = '<div class="error-msg">RiTa.js hasn\'t finished loading. Wait a moment and try again.</div>';
      return;
    }

    generateBtn.disabled = true;
    generateBtn.textContent = 'Weaving…';
    setWeaving(true);
    outputBody.innerHTML = '<p class="placeholder">Spinning your paragraph…</p>';

    setTimeout(() => {
      try {
        const n = parseInt(ngramSel.value, 10);
        const numSentences = parseInt(sentencesSel.value, 10);
        const rm = RiTa.markov(n);
        rm.addText(text);
        let result = rm.generate(numSentences);

        if (!result || (Array.isArray(result) && result.length === 0)){
          throw new Error('empty');
        }

        const paragraph = Array.isArray(result) ? result.join(' ') : String(result);
        renderWords(paragraph);
        outputMeta.textContent = numSentences + ' sentence' + (numSentences === 1 ? '' : 's') + ' · ' + n + '-gram';
        outputBody.querySelector('.output-text') && setupOutputActions(paragraph);
      } catch (err) {
        outputBody.innerHTML = '<p class="placeholder">Couldn\'t spin a paragraph from that text.<span class="hint">Try pasting a longer or more varied passage, or loosen the weave tightness.</span></p>';
        outputMeta.textContent = '';
      } finally {
        generateBtn.disabled = false;
        generateBtn.textContent = 'Generate paragraph';
        setWeaving(false);
      }
    }, 450);
  });

  function setupOutputActions(paragraph){
    const actions = document.createElement('div');
    actions.className = 'output-actions';
    const copyBtn = document.createElement('button');
    copyBtn.className = 'icon-btn';
    copyBtn.textContent = 'Copy text';
    copyBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(paragraph).then(() => {
        copyBtn.textContent = 'Copied';
        setTimeout(() => copyBtn.textContent = 'Copy text', 1400);
      });
    });
    const againBtn = document.createElement('button');
    againBtn.className = 'icon-btn';
    againBtn.textContent = 'Spin again';
    againBtn.addEventListener('click', () => generateBtn.click());
    actions.appendChild(copyBtn);
    actions.appendChild(againBtn);
    outputBody.appendChild(actions);
  }

  refreshState();
})();
