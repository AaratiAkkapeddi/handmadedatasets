const video = document.getElementById('camera');

function report(type, details = {}) {
  parent.postMessage({ type, ...details }, location.origin);
}

async function startTracking() {
  try {
    video.srcObject = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { width: 512, height: 512 }
    });
    await video.play();

    const handpose = await ml5.handpose(video);
    report('hand-tracker-ready');
    handpose.on('predict', function(predictions) {
      report('hand-tracker-predictions', {
        predictions,
        width: video.videoWidth,
        height: video.videoHeight
      });
    });
  } catch (error) {
    report('hand-tracker-error', { message: error.message || String(error) });
  }
}

startTracking();
