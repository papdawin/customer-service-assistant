from flask import Blueprint, render_template, request, jsonify, url_for, current_app
import os
import time
import wave
import librosa
import soundfile as sf
from werkzeug.utils import secure_filename
from .services import stt, llm, tts

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/transcribe', methods=['POST'])
def transcribe():
    total_start_time = time.time()

    if 'audio_data' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio_data']
    filename = secure_filename(audio_file.filename)
    upload_folder = current_app.config['UPLOAD_FOLDER']
    filepath = os.path.join(upload_folder, filename)
    audio_file.save(filepath)
    print(filepath)

    # Resample audio to 16kHz
    x, _ = librosa.load(filepath, sr=16000)
    sf.write(filepath, x, 16000)

    # Calculate audio duration
    with wave.open(filepath, 'rb') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)

    # Start total processing time measurement
    total_start = time.time()

    # Generate STT text
    transcription, proc_time = stt.transcribe_audio(filepath)
    print(transcription)
    print(type(transcription))
    print(f"Transcription Time: {proc_time:.2f} seconds")
    # Generate ChatGPT response
    chatgpt_reply, proc_time = llm.generate_response(transcription)
    print(f"ChatGPT Response Time: {proc_time:.2f} seconds")
    # Generate TTS audio
    proc_time = tts.generate_audio(chatgpt_reply)
    print(f"TTS Generation Time: {proc_time:.2f} seconds")
    # Total processing time
    total_time = time.time() - total_start

    # Output the timings
    print(f"Total Processing Time: {total_time:.2f} seconds")

    # Clean up uploaded file
    os.remove(filepath)

    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time

    audio_url = url_for('static', filename='output.wav')

    return jsonify({
        'transcription': transcription,
        'chatgpt_response': chatgpt_reply,
        'audio_url': audio_url,
        'audio_duration': f"{duration:.2f} seconds",
    })
