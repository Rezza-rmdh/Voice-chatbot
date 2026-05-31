import os
import uuid
import subprocess

from app.utils import generate_log

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TEMP_REMUX_DIR = os.path.join(BASE_DIR, "..", "temp", "remux")

os.makedirs(TEMP_REMUX_DIR, exist_ok = True)

def remux_audio(input_path: str) -> bytes | None:
    try:
        tmp_output_path = os.path.join(TEMP_REMUX_DIR, f"remux_{uuid.uuid4()}.wav")

        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            tmp_output_path
        ]

        result = subprocess.run(
            cmd,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
            text = True
        )

        if result.returncode != 0:
            generate_log(f"[ERROR] FFmpeg failed: {result.stderr}")
            return None

        with open(tmp_output_path, "rb") as f:
            remuxed_bytes = f.read()

        os.remove(tmp_output_path)

        return remuxed_bytes

    except Exception as e:
        generate_log(f"[ERROR] {str(e)}")
        return None