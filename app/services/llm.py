import time
import os
from llama_cpp import Llama

# Load the model only once globally
LLM_PATH = "/app/models/gemma-3-1b-it-UD-Q8_K_XL.gguf" # Docker
# LLM_PATH = "./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf" # Local
n_ctx = int(os.getenv("n_ctx", 2048))
n_threads = int(os.getenv("n_threads", 6))
n_gpu_layers = int(os.getenv("n_gpu_layers", 35))

llm = Llama(model_path=LLM_PATH, n_ctx=n_ctx, n_threads=n_threads, n_gpu_layers=n_gpu_layers)

def generate_response(transcription):
    start_time = time.time()
    try:
        prompt = (
            "You are a helpful assistant. Please answer as briefly as possible. "
            "Please don't use special characters in your answer. Please answer on the language the question was in.\n\n"
            f"User: {transcription}\nAssistant:"
        )

        response = llm(
            prompt,
            max_tokens=256,
            temperature=0.7,
            top_p=0.95,
            echo=False,
            stop=["User:", "Assistant:"]
        )
        chatgpt_reply = response["choices"][0]["text"].strip()
    except Exception as e:
        print("Error calling local LLM:", str(e))
        chatgpt_reply = "Error processing your request with the local LLM."
    end_time = time.time()
    processing_time = end_time - start_time
    return chatgpt_reply, processing_time
