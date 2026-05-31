import os
import uuid
import requests
import gradio as gr

API_URL = "http://127.0.0.1:8000/voice-chat"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "temp")

os.makedirs(TEMP_DIR, exist_ok=True)

def voice_chat(audio_path: str, mode: str):

    if not audio_path:
        return None, "Silakan rekam audio terlebih dahulu."

    try:
        with open(audio_path, "rb") as audio_file:

            response = requests.post(
                API_URL,
                files={
                    "file": (
                        os.path.basename(audio_path),
                        audio_file,
                        "audio/wav"
                    )
                },
                data={"mode": mode},
                timeout=300
            )

        if response.status_code != 200:
            return None, f"Backend Error {response.status_code}\n{response.text}"

        output_path = os.path.join(
            TEMP_DIR,
            f"response_{uuid.uuid4()}.wav"
        )

        with open(output_path, "wb") as f:
            f.write(response.content)

        return output_path, "Berhasil memproses audio."

    except Exception as e:
        return None, f"{str(e)}"


with gr.Blocks(
    title="Voice Chatbot",
    theme=gr.themes.Soft(primary_hue="orange")
) as demo:

    gr.Markdown(
        """
        # 🎙️ Voice Chatbot
        
        **Berbicara secara alami dan dapatkan jawaban suara dari AI.**
        """
    )

    mode_input = gr.Radio(
        choices=["normalize", "preserve"],
        value="normalize",
        label="Mode Percakapan"
    )

    audio_input = gr.Audio(
        sources=["microphone"],
        type="filepath",
        label="🎤 Rekam Pertanyaan Anda"
    )

    submit_btn = gr.Button(
        "Kirim Audio",
        variant="primary"
    )

    status_box = gr.Textbox(
        label="📋 Status",
        interactive=False,
        lines=3
    )

    audio_output = gr.Audio(
        type="filepath",
        label="🔊 Jawaban AI"
    )

    submit_btn.click(
        fn=voice_chat,
        inputs=[audio_input, mode_input],
        outputs=[audio_output, status_box]
    )

demo.launch()