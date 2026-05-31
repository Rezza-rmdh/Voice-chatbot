import os
import json
import time
import re
from datetime import datetime
from jiwer import wer, cer
from typing import Any
from app.stt import transcribe_speech_to_text
from app.llm import generate_response
from app.tts import transcribe_text_to_speech
from app.utils import generate_log
from utils.remux_audio import remux_audio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "data", "corpus", "audio")
REFERENCE_FILE = os.path.join(BASE_DIR, "data", "corpus", "transcripts", "reference.json")
RESULTS_DIR = os.path.join(BASE_DIR, "data", "results")
CHECKPOINT_FILE = os.path.join(RESULTS_DIR, "checkpoint.json")

os.makedirs(RESULTS_DIR, exist_ok = True)

def _load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return []

    try:
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []

def _save_checkpoint(results) :
    with open(CHECKPOINT_FILE, "w", encoding = "utf-8") as f:
        json.dump(results, f, ensure_ascii = False, indent = 4)    

def _load_reference() :
    with open(REFERENCE_FILE, "r", encoding = "utf-8") as f:
        return json.load(f)

def get_utterance_id(filename) :
    match = re.search(r"audio(\d+)", filename.lower())
    
    if match:
        raw_number = match.group(1)
        number = raw_number.lstrip("0") or "0"
        return f"audio{number}"
        
    return None

def _run_pipeline(audio_path, mode) :
    with open(audio_path, "rb") as f:
        file_bytes = f.read()
        
    t0 = time.time()
    transcript = transcribe_speech_to_text(file_bytes, ".wav")
    stt_latency = time.time() - t0

    if not transcript or "[ERROR]" in transcript:
        print(f"[WARN] Whisper STT Error detected. Attempting FFmpeg remux for: {os.path.basename(audio_path)}")
        generate_log(f"[WARN] Triggering remux for {audio_path} due to initial error: {transcript}")
        
        t_remux_start = time.time()
        remuxed_bytes = remux_audio(audio_path)
        
        if remuxed_bytes:
            new_transcript = transcribe_speech_to_text(remuxed_bytes, ".wav")
            stt_latency += (time.time() - t_remux_start)
            
            if "[ERROR]" not in new_transcript:
                print("[SUCCESS] Remux successful! Transcript recovered")
                transcript = new_transcript
            else:
                transcript = f"[ERROR] STT failed even after FFmpeg remux. Raw: {new_transcript}"
        else:
            transcript = "[ERROR] STT failed and FFmpeg remuxing system collapsed"

    if not transcript or "[ERROR]" in transcript:
        total_latency = time.time() - t0
        print("[SKIP] TTS error persists after remux, skipping text generation using LLM")
        
        return {
            "transcript": transcript.strip(),
            "response": "[ERROR] Maaf, terjadi gangguan pada sistem pemrosesan audio.",
            "audio_out": None,
            "stt_latency": round(stt_latency, 2),
            "llm_latency": 0.0,
            "tts_latency": 0.0,
            "total_latency": round(total_latency, 2),
            "is_failed": True
        }

    max_retries = 3
    base_delay = 2
    response_text = ""
    llm_latency = 0.0
    
    for attempt in range(max_retries):
        t1 = time.time()
        
        try:
            response_text = generate_response(transcript, mode)
            llm_latency = time.time() - t1
            
            if "[ERROR]" not in response_text and "500" not in response_text:
                break
        except Exception as api_err:
            response_text = f"[ERROR] Subprocess/API exception: {str(api_err)}"
            llm_latency = time.time() - t1

        if attempt < max_retries - 1:
            wait_time = base_delay * (2 ** attempt)
            print(f"[WARN] LLM Server Error (500). Retrying in {wait_time}s... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(wait_time)

    if "[ERROR]" in response_text or "500" in response_text:
        print("[SKIP] LLM error persisted after retries, skipping TTS synthesis")
        total_latency = time.time() - t0
        
        return {
            "transcript": transcript.strip(),
            "response": response_text,
            "audio_out": None,
            "mode": mode,
            "stt_latency": round(stt_latency, 2),
            "llm_latency": round(llm_latency, 2),
            "tts_latency": 0.0,
            "total_latency": round(total_latency, 2),
            "is_failed": True
        }
    
    t2 = time.time()
    audio_out = transcribe_text_to_speech(response_text)
    tts_latency = time.time() - t2

    total_latency = time.time() - t0

    return {
        "transcript": transcript.strip(),
        "response": response_text,
        "audio_out": audio_out,
        "mode": mode,
        "stt_latency": round(stt_latency, 2),
        "llm_latency": round(llm_latency, 2),
        "tts_latency": round(tts_latency, 2),
        "total_latency": round(total_latency, 2),
        "is_failed": False
    }

def main() :
    reference = _load_reference()
    raw_audio_files = sorted([f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")])
    audio_files = []
    
    for f in raw_audio_files:
        if re.search(r"\(\d+\)", f):
            print(f"[SKIP] Duplicate file skipped: {f}")
            continue
            
        audio_files.append(f)

    results = _load_checkpoint()
    processed = set((r["filename"], r["mode"]) for r in results if not r.get("is_failed", False))

    total_audio = len(audio_files)
    wer_scores, cer_scores = list(), list()

    print(f"[INFO] Total audio files: {total_audio}")

    for i, filename in enumerate(audio_files):
        audio_path = os.path.join(AUDIO_DIR, filename)
        utterance_id = get_utterance_id(filename)
        ref_text = reference.get(utterance_id, None)

        print("\n", end = None)
        print("[ FILE_INFO ]")
        print(f"Index        : {i + 1}/{total_audio}")
        print(f"Processing   : {filename}")
        print(f"Utterance ID : {utterance_id}")
        print(f"Reference    : {ref_text}")
            
        for j, mode in enumerate(["preserve", "normalize"]):
            print("\n", end = None)
            print(f"[ MODE: {mode} ]")
            
            if (filename, mode) in processed:
                print(f"[SKIP] File: {filename}, Mode: {mode} (already processed)")

                existing_record = next((r for r in results if r.get("filename") == filename and r.get("mode") == mode), None)

                if existing_record and existing_record.get("wer") is not None:
                    wer_scores.append(existing_record["wer"])
                    cer_scores.append(existing_record["cer"])

                continue
                
            print(f"File Name    : {filename}")
    
            try:
                result = _run_pipeline(audio_path, mode)
                transcript = result["transcript"]

                print("\n", end = None)
                print("[ RESULT ]")
                print(f"Transcript : {transcript}")
                print(f"Mode       : {mode}")
                print(f"Response   : {result['response']}")
                print(f"STT lat.   : {result['stt_latency']}s")
                print(f"LLM lat.   : {result['llm_latency']}s")
                print(f"TTS lat.   : {result['tts_latency']}s")
                print(f"Total lat. : {result['total_latency']}s")
                print(f"Is Failed  : {result['is_failed']}")
    
                if ref_text:
                    word_error = wer(ref_text.lower(), transcript.lower())
                    char_error = cer(ref_text.lower(), transcript.lower())
                    
                    print(f"WER        : {round(word_error, 4)}")
                    print(f"CER        : {round(char_error, 4)}")
    
                    wer_scores.append(word_error)
                    cer_scores.append(char_error)
                else:
                    word_error, char_error = None, None
                    log_message = f"[WARN] WER/CER: No reference found for {utterance_id}"
    
                    generate_log(log_message)
                    print(f"[WARN] {log_message}")
    
                results.append({
                    "filename": filename,
                    "utterance_id": utterance_id,
                    "mode": mode,
                    "reference": ref_text,
                    "transcript": transcript,
                    "response": result["response"],
                    "audio_output": result["audio_out"],
                    "wer": round(word_error, 4) if word_error is not None else None,
                    "cer": round(char_error, 4) if char_error is not None else None,
                    "stt_latency": result["stt_latency"],
                    "llm_latency": result["llm_latency"],
                    "tts_latency": result["tts_latency"],
                    "total_latency": result["total_latency"],
                    "is_failed": result["is_failed"]
                })
                
            except Exception as e:
                log_message = f"[ERROR] {e}"
    
                generate_log(log_message)
                print(f"[ERROR] {log_message}")
                
                results.append({
                    "filename": filename,
                    "utterance_id": utterance_id,
                    "mode": mode,
                    "reference": ref_text,
                    "error": str(e),
                    "is_failed": True
                })
    
            _save_checkpoint(results)
                
    print("\n", end = None)
    print("[ SUMMARY ]")
    
    avg_wer, avg_cer, avg_lat = None, None, None
    
    if wer_scores:
        avg_wer = round(sum(wer_scores) / len(wer_scores), 4)
        avg_cer = round(sum(cer_scores) / len(cer_scores), 4)
        
        print(f"Avg. WER       : {avg_wer}")
        print(f"Avg. CER       : {avg_cer}")

    latencies = [r["total_latency"] for r in results if "total_latency" in r]
    
    if latencies:
        avg_lat = round(sum(latencies) / len(latencies), 2)
        print(f"Avg. total lat.: {avg_lat}s")

    utterance_summary = {}
    
    for r in results:
        if "error" in r or "utterance_id" not in r:
            continue
            
        uid = r["utterance_id"]
        mode = r["mode"]
        
        if uid not in utterance_summary:
            utterance_summary[uid] = {}
            
        utterance_summary[uid][mode] = {
            "wer": r.get("wer"),
            "cer": r.get("cer"),
            "stt_latency": r.get("stt_latency"),
            "llm_latency": r.get("llm_latency"),
            "tts_latency": r.get("tts_latency"),
            "total_latency": r.get("total_latency"),
            "is_failed": r.get("is_failed")
        }
        
    output_file = os.path.join(RESULTS_DIR, f"pipeline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    with open(output_file, "w", encoding = "utf-8") as f:
        json.dump({
            "summary": {
                "total_files": len(audio_files),
                "avg_wer": avg_wer if wer_scores else None,
                "avg_cer": avg_cer if wer_scores else None,
                "avg_lat": avg_lat if latencies else None
            },
            "utterance_summary": utterance_summary,
            "results": results
        }, f, ensure_ascii = False, indent = 4)

    print("\n", end = f"[INFO] Results saved to: {output_file}")

if __name__ == "__main__":
    main()