import os
import json
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response

from app.stt import transcribe_speech_to_text
from app.tts import transcribe_text_to_speech
from app.llm import generate_response

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "log")
os.makedirs(LOG_DIR, exist_ok=True)

@app.post("/voice-chat")
async def voice_chat(
    file: UploadFile = File(...),
    mode: str = Form(default="normalize")
):  
    print(f"[DEBUG] Request masuk, file: {file.filename}, mode: {mode}")
    file_bytes = await file.read()
    print(f"[DEBUG] file_bytes size: {len(file_bytes)}")

    if len(file_bytes) < 1000:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Audio terlalu pendek atau kosong")
    
    file_ext = os.path.splitext(file.filename)[-1] or ".wav"

    #STT
    transcript = transcribe_speech_to_text(file_bytes, file_ext)

    #LLM
    response_text = generate_response(transcript, mode)
    if "[ERROR]" in response_text or "500 INTERNAL" in response_text:
        response_text = "Mohon maaf, server kecerdasan buatan sedang sibuk. Silakan coba beberapa saat lagi."

    #TTS
    audio_path = transcribe_text_to_speech(response_text)

    # Logging
    log_entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "transcript": transcript,
        "response": response_text,
        "audio_output": audio_path
    }

    log_path = os.path.join(LOG_DIR, f"log_{datetime.now().strftime('Y%m%d')}.json")

    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            logs = json.load(f)

    else:
        logs = []

    logs.append(log_entry)

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)

    with open(audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()

    print(f"[DEBUG] audio_path: {audio_path}")
    print(f"[DEBUG] audio_bytes size: {len(audio_bytes)}")

    return Response(content=audio_bytes, media_type="audio/wav")