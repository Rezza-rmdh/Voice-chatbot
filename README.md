# Voice Chatbot UAS – STT, Gemini LLM, TTS Integration

Proyek UAS ini merupakan aplikasi chatbot berbasis suara yang memungkinkan pengguna berbicara langsung melalui antarmuka web. Sistem akan mengenali suara pengguna, mengubahnya menjadi teks (Speech-to-Text), memprosesnya menggunakan model bahasa besar (Gemini API), lalu mengubah hasil jawabannya kembali menjadi suara (Text-to-Speech).

## Fitur Utama
- Speech-to-Text (STT) menggunakan `whisper.cpp` dari OpenAI.
- LLM Integration menggunakan Google Gemini API untuk menghasilkan respons dalam Bahasa Indonesia.
- Text-to-Speech (TTS) menggunakan model Coqui TTS (Indonesian TTS).
- Antarmuka pengguna interaktif berbasis `Gradio` untuk pengujian langsung dari browser.

## Struktur Proyek
```
voice_chatbot_project/
│
├── app/
│   ├── main.py            # Endpoint utama FastAPI
│   ├── llm.py             # Integrasi Gemini API
│   ├── stt.py             # Transkripsi suara (whisper.cpp)
│   ├── tts.py             # TTS dengan Coqui
│   └── coqui_utils/       # Model dan config Coqui TTS
│
├── gradio_app/
│   └── app.py             # Frontend dengan Gradio
│
├── models/
│   └── whisper.cpp/       # Hasil clone whisper.cpp
│
├── .env                   # Menyimpan Gemini API Key
├── requirements.txt       # Daftar dependensi Python
```

## Catatan
- Semua file audio menggunakan format `.wav`.
- Untuk menghasilkan fonem seperti `dəˈnɡan`, teks dari Gemini harus dikonversi ke fonetik.
- Disarankan menggunakan model Whisper: `ggml-small`.
- Gunakan speaker: `wibowo` dari model Coqui v1.2.

## Dibuat Untuk
Proyek UAS mata kuliah *Pemrosesan Bahasa Alami* — Semester Genap 2024/2025.