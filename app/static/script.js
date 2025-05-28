let mediaRecorder;
let audioChunks = [];

const recordButton = document.getElementById('record');
const stopButton = document.getElementById('stop');
const recordingStatus = document.getElementById('recording-status');
const transcription = document.getElementById('transcription');
const uploadForm = document.getElementById('upload-form');
const audioFileInput = document.getElementById('audio-file');

recordButton.addEventListener('click', async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    audioChunks = [];

    mediaRecorder.ondataavailable = event => {
        audioChunks.push(event.data);
    };

    mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        const formData = new FormData();
        formData.append('audio_data', audioBlob, 'recording.wav');

        recordingStatus.textContent = 'Transcribing...';

        const response = await fetch('/transcribe', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        transcription.textContent = result.transcription;
        recordingStatus.textContent = '';
    };

    mediaRecorder.start();
    recordButton.disabled = true;
    stopButton.disabled = false;
    recordingStatus.textContent = 'Recording...';
});

stopButton.addEventListener('click', () => {
    mediaRecorder.stop();
    recordButton.disabled = false;
    stopButton.disabled = true;
});

uploadForm.addEventListener('submit', async (event) => {
    event.preventDefault();

    const file = audioFileInput.files[0];
    if (!file) {
        alert('Please select a WAV file to upload.');
        return;
    }

    const formData = new FormData();
    formData.append('audio_data', file);

    transcription.textContent = 'Transcribing...';

    const response = await fetch('/transcribe', {
        method: 'POST',
        body: formData
    });

    const result = await response.json();
    transcription.textContent = result.transcription;
});
