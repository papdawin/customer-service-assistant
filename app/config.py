import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    STATIC_FOLDER = os.getenv('STATIC_FOLDER', 'static')
    DEBUG = os.getenv('DEBUG', 'False') == 'True'
