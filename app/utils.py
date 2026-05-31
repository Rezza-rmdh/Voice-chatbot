import os
import json
from datetime import datetime
from typing import Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "..", "app", "log")

os.makedirs(LOGS_DIR, exist_ok = True)

def generate_log(message: str) -> None:
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")
    log_message = f"[{time}] {message}"
    
    with open(os.path.join(LOGS_DIR, f"{date}.log"), "a") as log_file:
        log_file.write(log_message + "\n")

def read_instruction(path: str) -> str:
    with open(path, "r", encoding = "utf-8") as f:
        return f.read().strip()

def read_json(path: str) -> Any:
    with open(path, "r", encoding = "utf-8") as f:
        return json.load(f)