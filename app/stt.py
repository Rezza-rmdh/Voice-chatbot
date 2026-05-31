import os
import uuid
import subprocess

from app.utils import generate_log

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

WHISPER_DIR = os.path.join(BASE_DIR, "..", "models", "whisper.cpp")
WHISPER_BINARY = os.path.join(WHISPER_DIR, "build", "bin", "Release", "whisper-cli")
WHISPER_MODEL_PATH = os.path.join(WHISPER_DIR, "models", "ggml-small.bin")

TEMP_STT_DIR = os.path.join(BASE_DIR, "..", "temp", "stt")

os.makedirs(TEMP_STT_DIR, exist_ok=True)

def transcribe_speech_to_text(file_bytes: bytes, file_ext: str = ".wav") -> str:
    request_id = str(uuid.uuid4())

    audio_path = os.path.join(TEMP_STT_DIR, f"{request_id}{file_ext}")
    transcription_prefix = os.path.join(TEMP_STT_DIR, request_id)
    result_path = f"{transcription_prefix}.txt"

    with open(audio_path, "wb") as f:
        f.write(file_bytes)

    cmd = [
        WHISPER_BINARY,
        "-m", WHISPER_MODEL_PATH,
        "-f", audio_path,
        "-l", "id",
        "-otxt",
        "-of", transcription_prefix
    ]

    try:
        subprocess.run(cmd, check = True)
        
    except subprocess.CalledProcessError as e:
        log_message = f"[ERROR] Whisper failed: {e}"
        generate_log(log_message)

        return log_message
    
    try:
        with open(result_path, "r", encoding = "utf-8") as result_file:
            transcript = result_file.read()

        return transcript
    
    except FileNotFoundError:
        log_message = "[ERROR] Transcription file not found"
        generate_log(log_message)

        return log_message