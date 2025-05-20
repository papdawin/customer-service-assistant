import time
from openai import OpenAI
from flask import current_app

def generate_response(transcription):
    start_time = time.time()
    try:
        client = OpenAI(api_key=current_app.config['OPENAI_API_KEY'])
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Please answer as briefly as possible. Please don't use special characters in your answer."
                },
                {
                    "role": "user",
                    "content": transcription
                }
            ]
        )
        chatgpt_reply = response.choices[0].message.content
    except Exception as e:
        print("Error calling OpenAI API:", str(e))
        chatgpt_reply = "Error processing your request with ChatGPT."
    end_time = time.time()
    processing_time = end_time - start_time
    return chatgpt_reply, processing_time
