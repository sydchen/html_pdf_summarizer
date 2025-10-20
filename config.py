import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuration settings for the summarizer application"""

    # Ollama Model
    MODEL = os.getenv('MODEL', 'gemma3:4b')

    # Whisper Configuration
    WHISPER_MODEL_PATH = os.getenv(
        'WHISPER_MODEL_PATH',
        '/Users/sydchen/projects/asr/whisper.cpp/models/ggml-medium.bin'
    )
    WHISPER_BINARY_PATH = os.getenv(
        'WHISPER_BINARY_PATH',
        '/Users/sydchen/projects/asr/whisper.cpp/build/bin/whisper-cli'
    )
    WHISPER_LANGUAGE = os.getenv('WHISPER_LANGUAGE', 'auto')

    # Language code mapping for Whisper
    # Maps common language codes to Whisper-supported codes
    LANGUAGE_MAP = {
        'zh': 'zh',      # Chinese
        'zh-CN': 'zh',   # Simplified Chinese
        'zh-TW': 'zh',   # Traditional Chinese
        'ja': 'ja',      # Japanese
        'en': 'en',      # English
        'ko': 'ko',      # Korean
        'es': 'es',      # Spanish
        'fr': 'fr',      # French
        'de': 'de',      # German
        'pt': 'pt',      # Portuguese
        'ru': 'ru',      # Russian
        'ar': 'ar',      # Arabic
        'hi': 'hi',      # Hindi
    }

    # YouTube Download Settings
    YOUTUBE_OUTPUT_DIR = os.getenv('YOUTUBE_OUTPUT_DIR', './youtube_downloads')
    KEEP_AUDIO_FILES = os.getenv('KEEP_AUDIO_FILES', 'true').lower() == 'true'
    KEEP_TRANSCRIPT_FILES = os.getenv('KEEP_TRANSCRIPT_FILES', 'true').lower() == 'true'

    # Audio Processing
    AUDIO_SAMPLE_RATE = int(os.getenv('AUDIO_SAMPLE_RATE', '16000'))
    AUDIO_CHANNELS = int(os.getenv('AUDIO_CHANNELS', '1'))

    @classmethod
    def ensure_output_dir(cls):
        """Ensure the output directory exists"""
        os.makedirs(cls.YOUTUBE_OUTPUT_DIR, exist_ok=True)
        return cls.YOUTUBE_OUTPUT_DIR
