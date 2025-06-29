import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import time

# model_id = "openai/whisper-large-v3"
# model_id = "sarpba/whisper-hu-tiny-finetuned-V2"
model_id = "sarpba/whisper-hu-large-v3-turbo-finetuned"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print(f"STT is running on: {device}")

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    torch_dtype=torch_dtype,
    device=device,
)

def transcribe_audio(filepath):
    start_time = time.time()
    result = pipe(filepath, return_timestamps=True)
    end_time = time.time()
    transcription = result['text']
    processing_time = end_time - start_time
    return transcription, processing_time
