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

    const facemesh = await ml5.facemesh(video);
    report('face-tracker-ready');
    facemesh.on('face', function(predictions) {
      report('face-tracker-predictions', {
        predictions,
        width: video.videoWidth,
        height: video.videoHeight
      });
    });
  } catch (error) {
    report('face-tracker-error', { message: error.message || String(error) });
  }
}

startTracking();
