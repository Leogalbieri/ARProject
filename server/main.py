import socket
import msgpack
import struct
import torch
import numpy as np
import cv2
from dotenv import load_dotenv
import os
import config

load_dotenv()
PORT = os.getenv("PORT")

if not PORT:
    raise ValueError("PORT not set in .env")

PORT = int(PORT)

# --- Device ---
if config.DEVICE is None:
    config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {config.DEVICE}")

# --- Models ---
from models.primary import PrimaryModel
from models.secondary import SecondaryModel
from models.search_model import SearchModel

primary = PrimaryModel()
secondary = SecondaryModel()
search = SearchModel()

# --- Setup server ---
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("0.0.0.0", PORT))
server.listen(1)

# --- Warmup ---
dummy = np.zeros((480, 640, 3), dtype=np.uint8)
primary.infer(dummy)
print("Ready!")

payload_size = struct.calcsize("Q")

from modes import general, search as search_mode

def recv_all(conn, size):
    buffer = b""
    while len(buffer) < size:
        packet = conn.recv(size - len(buffer))
        if not packet:
            return None
        buffer += packet
    return buffer

while True:
    print("Waiting for connection...")
    conn, addr = server.accept()
    print(f"Connected: {addr}")

    try:
        while True:
            # --- Receive header ---
            header = recv_all(conn, payload_size)
            if header is None:
                print("Client disconnected")
                break

            msg_size = struct.unpack("Q", header)[0]

            # --- Receive payload ---
            payload_data = recv_all(conn, msg_size)
            if payload_data is None:
                print("Client disconnected")
                break

            payload = msgpack.unpackb(payload_data, raw=True)

            # Validate frame exists
            if b"frame" not in payload:
                continue

            frame = cv2.imdecode(np.frombuffer(payload[b"frame"], np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            mode = payload.get(b"mode", b"general").decode()

            # --- Inference ---
            if mode == "general":
                annotated = general.run(frame, primary, secondary)
            elif mode == "search":
                annotated = search_mode.run(frame, search)
            else:
                annotated = frame

            # --- Encode and send back ---
            _, buffer = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            reply = msgpack.packb(buffer.tobytes())
            conn.sendall(struct.pack("Q", len(reply)) + reply)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        conn.close()
        print("Connection closed")
