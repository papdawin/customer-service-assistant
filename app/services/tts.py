import time
from TTS.api import TTS
import torch

tts_model = TTS("tts_models/hu/css10/vits")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# tts_model.to("cuda" if torch.cuda.is_available() else "cpu")
tts_model.to(device)
print(f"TTS is running on: {device}")

def generate_audio(text, output_path="./app/static/output.wav"):
    start_time = time.time()
    tts_model.tts_to_file(text=text, file_path=output_path)
    end_time = time.time()
    processing_time = end_time - start_time
    return processing_time
