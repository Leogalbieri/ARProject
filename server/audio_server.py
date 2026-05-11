from faster_whisper import WhisperModel
import socket
import sounddevice as sd
import numpy as np
import os
from dotenv import load_dotenv
import config
from intent_recognition import process

load_dotenv()
PORT = os.getenv("AUDIO_PORT")

if not PORT:
    raise ValueError("PORT not set in .env")

PORT = int(PORT)

# Setup server
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("0.0.0.0", PORT))
server.listen(1)

SAMPLERATE = config.AUDIO_SAMPLERATE
CHANNELS = config.AUDIO_CHANNELS
DATA_TYPE = config.AUDIO_DATA_TYPE

model = WhisperModel(config.WHISPER_SIZE, device=config.WHISPER_DEVICE, compute_type=config.WHISPER_COMPUTE_TYPE)


while True:
    print("Waiting for connection...")
    conn, addr = server.accept()
    print(f"Connected to {addr}")

    # Reset audio buffer
    audio_bytes = b''

    try:
        while True:
            size_data = conn.recv(4)
            if not size_data:
                break

            chunk_size = int.from_bytes(size_data, 'big')

            if chunk_size == 0:
                break

            chunk = b''
            while len(chunk) < chunk_size:
                packet = conn.recv(chunk_size - len(chunk))
                if not packet:
                    break
                chunk += packet

            audio_bytes += chunk

    except Exception as e:
        print(f"Error: {e}")

    finally:
        conn.close()
        print("Connection closed")

    audio_array = np.frombuffer(audio_bytes, dtype=DATA_TYPE)
    audio_array = audio_array.astype(np.float32) / 32768.0


    def transcribe(audio_array):
        segments, info = model.transcribe(
            audio_array,
            beam_size=config.WHISPER_BEAM_SIZE,
            language=config.WHISPER_LANGUAGE,
            vad_filter=config.WHISPER_VAD_FILTER
        )
        return "".join(segment.text for segment in segments)

    text = transcribe(audio_array)
    process(text)

    sd.wait()
