import os
from google import genai
from google.genai import types
from pydantic import TypeAdapter
from dotenv import load_dotenv

load_dotenv()

MODEL = "gemma-4-26b-a4b-it"

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHAT_HISTORY_FILE = os.path.join(BASE_DIR, "chat_history.json")

# Prompt sistem yang digunakan untuk membimbing gaya respons LLM
system_instruction = """
You are a responsive, intelligent, and fluent virtual assistant who communicates in Indonesian.
Your task is to provide clear, concise, and informative answers in response to user queries or statements spoken through voice.

Your answers must:
- Be written in polite and easily understandable Indonesian.
- Be short and to the point (maximum 2–3 sentences).
- Avoid repeating the user's question; respond directly with the answer.

Example tone:
User: Cuaca hari ini gimana?
Assistant: Hari ini cuacanya cerah di sebagian besar wilayah, dengan suhu sekitar 30 derajat.

User: Kamu tahu siapa presiden Indonesia?
Assistant: Presiden Indonesia saat ini adalah Joko Widodo.

If you're unsure about an answer, be honest and say that you don't know.
"""

# TODO: Inisialisasi klien Gemini dan konfigurasi prompt
# Gunakan genai.Client(api_key=...) untuk membuat klien.
# Gunakan types.GenerateContentConfig(system_instruction=...) untuk membuat konfigurasi awal.
# Jika ingin melihat contoh implementasi, baca dokumentasi resmi Gemini:
# https://github.com/google-gemini/cookbook/blob/main/quickstarts/Get_started.ipynb
client = genai.Client(api_key = GOOGLE_API_KEY)
chat_config = ...
history_adapter = TypeAdapter(list[types.Content])

# Fungsi untuk menyimpan/memuat riwayat chat
def export_chat_history(chat) -> str:
    return history_adapter.dump_json(chat.get_history()).decode("utf-8")

def save_chat_history(chat):
    json_history = export_chat_history(chat)
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
        f.write(json_history)

def load_chat_history():
    if not os.path.exists(CHAT_HISTORY_FILE):
        return client.chats.create(model=MODEL, config=chat_config)
    
    if os.path.getsize(CHAT_HISTORY_FILE) == 0:
        return client.chats.create(model=MODEL, config=chat_config)

    with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
        json_str = f.read().strip()

    if not json_str:
        return client.chats.create(model=MODEL, config=chat_config)

    try:
        history = history_adapter.validate_json(json_str)
        return client.chats.create(model=MODEL, config=chat_config, history=history)
    except Exception as e:
        print(f"[ERROR] Gagal load history chat: {e}")
        return client.chats.create(model=MODEL, config=chat_config)

# Inisialisasi sesi chat saat aplikasi dimulai
chat = load_chat_history()

# Kirim prompt ke LLM dan kembalikan respons teks
def generate_response(prompt: str) -> str:
    try:
        response = chat.send_message(prompt)
        save_chat_history(chat)
        return response.text.strip()
    except Exception as e:
        return f"[ERROR] {str(e)}"
