import os
from google import genai
from google.genai import types
from google.genai.types import HttpOptions
from pydantic import TypeAdapter
from dotenv import load_dotenv
from app.utils import generate_log, read_instruction
from typing import Any

load_dotenv()

MODEL = "gemma-4-26b-a4b-it"

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_DIR = os.path.join(BASE_DIR, "..", "data", "history")
os.makedirs(HISTORY_DIR, exist_ok = True)

CHAT_HISTORY_FILE = os.path.join(HISTORY_DIR, "chat_history.json")
PROMPT_DIR = os.path.join(BASE_DIR, "..", "data", "prompt")

MAX_HISTORY_MESSAGES = 50
RESPONSE_TIMEOUT = 3 * (60 * 1000)

preserve = read_instruction(os.path.join(PROMPT_DIR, "preserve_mode.txt"))
normalize = read_instruction(os.path.join(PROMPT_DIR, "normalize_mode.txt"))

client = genai.Client(api_key = GOOGLE_API_KEY, http_options = HttpOptions(timeout = RESPONSE_TIMEOUT))
history_adapter = TypeAdapter(list[types.Content])

def _export_chat_history(chat) -> str:
    raw_history = chat.get_history()
    cleaned_history: list = list()
    
    for content in raw_history:
        cleaned_parts: list = list()
        
        for part in content.parts:
            if hasattr(part, "thought") and part.thought:
                continue
                
            cleaned_parts.append(part)
        
        if cleaned_parts:
            cleaned_content = types.Content(role = content.role, parts = cleaned_parts)
            cleaned_history.append(cleaned_content)

    if len(cleaned_history) > MAX_HISTORY_MESSAGES:
        start_index = len(cleaned_history) - MAX_HISTORY_MESSAGES
        
        if cleaned_history[start_index].role == "model" or cleaned_history[start_index].role == "assistant":
            start_index += 1
            
        cleaned_history = cleaned_history[start_index:]
            
    return history_adapter.dump_json(cleaned_history).decode("utf-8")

def _save_chat_history(chat) -> None:
    json_history = _export_chat_history(chat)
    
    with open(CHAT_HISTORY_FILE, "w", encoding = "utf-8") as f:
        f.write(json_history)

def _load_chat_history(config: str) -> Any:
    if not os.path.exists(CHAT_HISTORY_FILE):
        return client.chats.create(model = MODEL, config = config)
    
    if os.path.getsize(CHAT_HISTORY_FILE) == 0:
        return client.chats.create(model = MODEL, config = config)

    with open(CHAT_HISTORY_FILE, "r", encoding = "utf-8") as f:
        json_str = f.read().strip()

    if not json_str:
        return client.chats.create(model = MODEL, config = config)

    try:
        history = history_adapter.validate_json(json_str)

        if len(history) > MAX_HISTORY_MESSAGES:
            start_index = len(history) - MAX_HISTORY_MESSAGES
            
            if history[start_index].role == "model":
                start_index += 1
                
            history = history[start_index:]
        
        return client.chats.create(model = MODEL, config = config, history = history)
    except Exception as e:
        log_message = f"[ERROR] Failed to load chat history: {e}"
        print(log_message)
        generate_log(log_message)
        
        return client.chats.create(model = MODEL, config = config)
   
def _count_tokens(client: Any, history: Any | None = None, model: str | None = None) -> str:
    total_messages = len(history) if history else 0
    history_count = max(0, total_messages - 1)
    token_info = client.models.count_tokens(model = model, contents = history)
    
    return f"[INFO] {token_info} total_history_messages={history_count}"
    
def generate_response(prompt: str, mode: str = "normalize") -> str:
    if mode == "preserve":
        config = types.GenerateContentConfig(system_instruction = preserve)
    else:
        config = types.GenerateContentConfig(system_instruction = normalize)

    try:
        chat = _load_chat_history(config)
        extra = types.UserContent(parts = [types.Part(text = prompt)])
        history = [*chat.get_history(), extra]
        
        print(_count_tokens(client, history, MODEL))
        
        response = chat.send_message(prompt)
        final_text: str = str()
        
        if response.candidates and response.candidates[0].content.parts:
            text_parts = [
                part.text for part in response.candidates[0].content.parts 
                if not (hasattr(part, "thought") and part.thought) and part.text
            ]
            final_text = "".join(text_parts).strip()
        
        if not final_text:
            final_text = response.text.strip() if response.text else ""
            
        print(f"[INFO] LLM Response: \"{final_text}\"")
        _save_chat_history(chat)
        
        return final_text
    except Exception as e:
        log_message = f"[ERROR] {str(e)}"
        generate_log(log_message)
        
        return log_message