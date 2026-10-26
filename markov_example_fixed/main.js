(function () {
  const generateBtn = document.getElementById('generate-btn');
  const errorBox = document.getElementById('error-box');
  const outputBody = document.getElementById('output-body');
  const wordCount = document.getElementById('word-count');

  let corpusText = '';

  function setError(message) {
    errorBox.innerHTML = `<div class="error-msg">${message}</div>`;
  }

  async function loadCorpus() {
    try {
      const response = await fetch('./craigslist_missed.txt');

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      corpusText = await response.text();
      const words = corpusText.trim().split(/\s+/).filter(Boolean);
      wordCount.textContent = ``;

      if (words.length < 25) {
        throw new Error('The corpus is too short.');
      }

      generateBtn.disabled = false;
    } catch (err) {
      console.error(err);
      wordCount.textContent = 'Corpus unavailable';
      setError('Could not load ./craigslist_missed.txt.');
      generateBtn.disabled = true;
    }
  }

  generateBtn.addEventListener('click', () => {
    errorBox.innerHTML = '';

    if (!corpusText) {
      setError('The corpus has not loaded yet.');
      return;
    }

    if (typeof RiTa === 'undefined') {
      setError('RiTa.js could not be loaded.');
      return;
    }

    generateBtn.disabled = true;

    try {
      const markov = RiTa.markov(3);
      markov.addText(corpusText);

      const generated = markov.generate(3);
      const paragraph = Array.isArray(generated)
        ? generated.join(' ')
        : String(generated);

      outputBody.textContent = paragraph;
    } catch (err) {
      console.error(err);
      setError('Could not generate text from the corpus.');
    } finally {
      generateBtn.disabled = false;
    }
  });

  loadCorpus();
})();
