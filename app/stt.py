import os
import uuid
import tempfile
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

WHISPER_DIR = os.path.join(BASE_DIR , "..", "models", "whisper.cpp")
WHISPER_BINARY = os.path.join(WHISPER_DIR, "build", "bin", "Release", "whisper-cli.exe")
WHISPER_MODEL_PATH = os.path.join(WHISPER_DIR, "models", "ggml-small.bin")

def transcribe_speech_to_text(file_bytes: bytes, file_ext: str = ".wav") -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, f"{uuid.uuid4()}{file_ext}")
        result_path = os.path.join(tmpdir, "transcription.txt")

        # simpan audio ke file temporer
        with open(audio_path, "wb") as f:
            f.write(file_bytes)

        # jalankan whisper.cpp dengan subprocess
        cmd = [
            WHISPER_BINARY,
            "-m", WHISPER_MODEL_PATH,
            "-f", audio_path,
            "-otxt",
            "-of", os.path.join(tmpdir, "..", "transcription")
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            return f"[ERROR] Whisper failed: {e}"
        
        # baca hasil transkripsi
        try:
            with open(result_path, "r", encoding="utf-8") as result_file:
                return result_file.read()
        except FileNotFoundError:
            return "[ERROR] Transcription file not found"
