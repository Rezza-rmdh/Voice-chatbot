import os
import uuid
import subprocess
import re
from app.utils import generate_log, read_json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

COQUI_DIR = os.path.join(BASE_DIR, "coqui_tts")
COQUI_MODEL_PATH = os.path.join(COQUI_DIR, "checkpoint_1260000-inference.pth")
COQUI_CONFIG_PATH = os.path.join(COQUI_DIR, "config.json")
COQUI_SPEAKER = "wibowo"
TEMP_TTS_DIR = os.path.join(BASE_DIR, "..", "temp", "tts")

os.makedirs(TEMP_TTS_DIR, exist_ok=True)

CORPUS_DIR = os.path.join(BASE_DIR, "..", "data", "corpus", "transcripts")
ALPHABET_PHONETIC = read_json(os.path.join(CORPUS_DIR, "alpha_phonetic.json"))
NUMERIC_PHONETIC = read_json(os.path.join(CORPUS_DIR, "num_phonetic.json"))
ENGLISH_PHONETIC = read_json(os.path.join(CORPUS_DIR, "english_phonetic.json"))
ARABIC_PHONETIC = read_json(os.path.join(CORPUS_DIR, "arabic_phonetic.json"))
PHONEME_MAP = read_json(os.path.join(CORPUS_DIR, "phoneme.json"))

def _preprocess_acronyms_and_numbers(text: str) -> str:
    def expand_match(match):
        token = match.group(0)
        expanded = []

        for char in token:
            lower = char.lower()

            if lower in ALPHABET_PHONETIC:
                expanded.append(ALPHABET_PHONETIC[lower])
            elif lower in NUMERIC_PHONETIC:
                expanded.append(NUMERIC_PHONETIC[lower])
            else:
                expanded.append(char)

        return " ".join(expanded)

    text = re.sub(r"\b[A-Z0-9]{2,}\b", expand_match, text)

    return text

def _remove_tags(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[[^\]]+\]", " ", text)
    text = re.sub(r"\([^)]+\)", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def _normalize_articulation(text: str) -> str:
    text=text.lower()
    text=re.sub(r"v", "f", text)
    text=re.sub(r"d\b", "t", text)
    text=re.sub(r"b\b", "p", text)
    text=re.sub(r"g\b", "k", text)
    text=text.replace("-", " ")
    text=re.sub(r"\s+", " ", text)

    return text.strip()

def _apply_english_phonetics(text: str) -> str:
    words = text.split()
    result = []

    for word in words:
        clean = re.sub(r"[^\w\-]", "", word.lower())
        punctuation = re.sub(r"[\w\-]", "", word)

        if clean in ENGLISH_PHONETIC:
            result.append(
                ENGLISH_PHONETIC[clean] + punctuation
            )
        else:
            result.append(word)

    return " ".join(result)

def _apply_arabic_phonetics(text: str) -> str:
    words = text.split()
    result = []

    for word in words:
        clean = re.sub(r"[^\w\-]", "", word.lower())
        punctuation = re.sub(r"[\w\-]", "", word)

        if clean in ARABIC_PHONETIC:
            result.append(
                ARABIC_PHONETIC[clean] + punctuation
            )
        else:
            result.append(word)

    return " ".join(result)

def _grapheme_to_phoneme(text: str) -> str:
    sorted_keys = sorted(PHONEME_MAP.keys(),key=len,reverse=True)
    pattern = re.compile("|".join(map(re.escape, sorted_keys)))
    words = text.split()
    result = []

    for word in words:
        clean = re.sub(r"[^\w]", "", word)
        punctuation = re.sub(r"[\w]", "", word)
        converted = pattern.sub(lambda m: PHONEME_MAP[m.group()],clean)
        result.append(converted + punctuation)

    return " ".join(result)

def transcribe_text_to_speech(text: str) -> str:
    if not text:
        return "[ERROR] Empty text"

    processed_text = _remove_tags(text)
    processed_text = _preprocess_acronyms_and_numbers(processed_text)
    processed_text = _apply_english_phonetics(processed_text)
    processed_text = _apply_arabic_phonetics(processed_text)
    normalized_text = _normalize_articulation(processed_text)
    final_processed_text = _grapheme_to_phoneme(normalized_text)

    print(f"[INFO] TTS INPUT: {final_processed_text}")

    path = _tts_with_coqui(final_processed_text)

    return path

def _tts_with_coqui(text: str) -> str:
    output_path = os.path.join(TEMP_TTS_DIR, f"tts_{uuid.uuid4()}.wav")

    cmd = [
        "tts",
        "--text", text,
        "--model_path", COQUI_MODEL_PATH,
        "--config_path", COQUI_CONFIG_PATH,
        "--speaker_idx", COQUI_SPEAKER,
        "--out_path", output_path
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        log_message = f"[ERROR] TTS subprocess failed: {e}"
        print(log_message)
        generate_log(log_message)

        return "[ERROR] Failed to synthesize speech"

    return output_path