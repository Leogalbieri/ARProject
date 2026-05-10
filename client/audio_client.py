import socket
import sounddevice as sd
import numpy as np
from openwakeword import Model
import webrtcvad
import os
from dotenv import load_dotenv
import config

load_dotenv()
PC_IP = os.getenv("PC_IP")
PORT = os.getenv("AUDIO_PORT")

if not PC_IP:
    raise ValueError("PC_IP not set or empty in .env")

if not PORT:
    raise ValueError("PORT not set in .env")

try:
    socket.inet_aton(PC_IP)
except socket.error:
    raise ValueError(f"Invalid PC_IP: {PC_IP}")

PORT = int(PORT)

SAMPLERATE = config.AUDIO_SAMPLERATE
CHANNELS = config.AUDIO_CHANNELS
DATA_TYPE = config.AUDIO_DATA_TYPE
CHUNK_SIZE = config.AUDIO_CHUNK_SIZE

VAD_FRAME = config.VAD_FRAME
FRAME_SAMPLES = int(SAMPLERATE * VAD_FRAME / 1000)  # 480 samples

WAKE_WORD_THRESHOLD = config.WAKE_WORD_THRESHOLD
SILENCE_FRAMES_LIMIT = config.SILENCE_FRAMES_LIMIT   # seconds of silence to stop recording
MAX_DURATION = int((config.MAX_DURATION * 1000) / VAD_FRAME)   # seconds max of recording

OWW_MODEL = Model(
    wakeword_models=[config.WAKE_WORD_MODEL],
    inference_framework=config.WAKE_WORD_INFERENCE_FRAMEWORK)

VAD = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)  # Aggressiveness


def send_audio(audio_bytes):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((PC_IP, PORT))
    print(f"Connected to server at {PC_IP} : {PORT}")

    offset = 0
    while offset < len(audio_bytes):
        chunk = audio_bytes[offset:offset + CHUNK_SIZE]
        client.sendall(len(chunk).to_bytes(4, "big"))
        client.sendall(chunk)
        offset += CHUNK_SIZE

    client.sendall((0).to_bytes(4, "big"))
    client.close()


print("Lisntening for wake word...")

audio_buffer = np.zeros(0, dtype=DATA_TYPE)
recording = False
recorded_frames = []
silence_count = 0


def audio_callback(indata, frames, time, status):
    global recording, recorded_frames, silence_count, audio_buffer

    frame = indata[:, 0].copy()
    audio_buffer = np.append(audio_buffer, frame)

    oww_chunk_size = 1280
    while not recording and len(audio_buffer) >= oww_chunk_size:
        chunk = audio_buffer[:oww_chunk_size]
        audio_buffer = audio_buffer[oww_chunk_size:]

        prediction = OWW_MODEL.predict(chunk)
        score = list(prediction.values())[0]

        if score > WAKE_WORD_THRESHOLD:
            print(f"Wake Word detected! (score: {score:.2f}) - Listening...")
            recording = True
            recorded_frames = []
            silence_count = 0

    if recording:
        recorded_frames.append(frame.copy())

        # VAD checking for silence
        frame_bytes = frame[:FRAME_SAMPLES].astype(DATA_TYPE).tobytes()
        if len(frame_bytes) == FRAME_SAMPLES * 2:
            is_speech = VAD.is_speech(frame_bytes, SAMPLERATE)
            if not is_speech:
                silence_count += 1
            else:
                silence_count = 0  # Resets on speech

            if silence_count >= SILENCE_FRAMES_LIMIT or len(recorded_frames) >= MAX_DURATION:
                recording = False

                if silence_count >= SILENCE_FRAMES_LIMIT:
                    print("End of phrase detected, sending audio to server...")
                else:
                    print("Timed out, sending audio to server...")

                audio_data = np.concatenate(recorded_frames)
                send_audio(audio_data.tobytes())

                OWW_MODEL.reset()   # Resets oww model to avoid loops

                print("Listening for wake word...")


with sd.InputStream(
    samplerate=SAMPLERATE,
    channels=CHANNELS,
    dtype=DATA_TYPE,
    blocksize=FRAME_SAMPLES,
    callback=audio_callback
):
    while True:
        sd.sleep(100)
