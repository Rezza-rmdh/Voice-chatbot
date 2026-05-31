import os
import uuid
import time

from typing import Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse

from app.stt import transcribe_speech_to_text
from app.tts import transcribe_text_to_speech
from app.llm import generate_response
from app.utils import generate_log

from utils.remux_audio import remux_audio

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_REMUX_DIR = os.path.join(BASE_DIR, "..", "temp", "remux")

os.makedirs(TEMP_REMUX_DIR, exist_ok = True)

@app.post("/voice-chat")
async def voice_chat(file: UploadFile = File(...), mode: str = Form("normalize")) -> Any:
    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code = 400, detail = "Audio kosong")

    file_ext = os.path.splitext(file.filename)[-1] or ".wav"

    # STT
    transcript = transcribe_speech_to_text(file_bytes, file_ext)

    if "[ERROR]" in transcript:
        generate_log(f"[WARN] STT failed. Attempting remux for {file.filename}")
        temp_path = os.path.join(TEMP_REMUX_DIR, f"input_{uuid.uuid4()}{file_ext}")
        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        try:
            remuxed_bytes = remux_audio(temp_path)
            if remuxed_bytes:
                transcript = transcribe_speech_to_text(remuxed_bytes, ".wav")

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    if "[ERROR]" in transcript:
        raise HTTPException(status_code = 400, detail = "Audio tidak dapat diproses")

    print(f"[INFO] STT Transcript: {transcript}")

    # LLM
    max_retries = 3
    base_delay = 2
    response_text = ""

    for attempt in range(max_retries):
        try:
            response_text = generate_response(transcript, mode)

            if "[ERROR]" not in response_text and "500" not in response_text:
                break

        except Exception as e:
            response_text = f"[ERROR] {str(e)}"

        if attempt < max_retries - 1:
            wait_time = base_delay * (2 ** attempt)
            generate_log(
                f"[WARN] LLM retry in {wait_time}s "
                f"(Attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(wait_time)

    if "[ERROR]" in response_text or "500" in response_text:
        raise HTTPException(status_code = 500, detail = f"LLM failed after {max_retries} retries")

    print(f"[INFO] LLM Response: {response_text}")

    # TTS
    audio_path = transcribe_text_to_speech(response_text)

    if "[ERROR]" in audio_path:
        raise HTTPException(status_code = 500, detail = audio_path)
    if not os.path.exists(audio_path):
        raise HTTPException(status_code = 500, detail = "Audio output missing")

    generate_log(f"[SUCCESS] Request completed ({mode})")

    return FileResponse(audio_path,media_type = "audio/wav",filename = "response.wav")