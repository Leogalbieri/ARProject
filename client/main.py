import cv2
import socket
import msgpack
import struct
import time
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
PC_IP = os.getenv("PC_IP")
PORT = os.getenv("PORT")

if not PC_IP:
    raise ValueError("PC_IP not set or empty in .env")

if not PORT:
    raise ValueError("PORT not set in .env")

try:
    socket.inet_aton(PC_IP)
except socket.error:
    raise ValueError(f"Invalid PC_IP: {PC_IP}")

PORT = int(PORT)

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((PC_IP, PORT))
print(f"Connected to server at {PC_IP}:{PORT}")
time.sleep(1)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

payload_size = struct.calcsize("Q")

def recv_all(sock, size):
    buffer = b""
    while len(buffer) < size:
        packet = sock.recv(size - len(buffer))
        if not packet:
            return None
        buffer += packet
    return buffer

while True:
    start = time.time()

    ret, frame = cap.read()
    if not ret:
        print("Error: Camera frame not found")
        continue

    # JPEG compression
    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    payload = msgpack.packb({
        "frame": buffer.tobytes(),
    })

    try:
        client.sendall(struct.pack("Q", len(payload)) + payload)
    except:
        print("Connection lost")
        break

    # --- Receive response ---
    header = recv_all(client, payload_size)
    if header is None:
        break

    msg_size = struct.unpack("Q", header)[0]

    frame_data = recv_all(client, msg_size)
    if frame_data is None:
        break

    encoded = msgpack.unpackb(frame_data, raw=True)

    annotated = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)

    cv2.imshow("AR View", annotated)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # FPS control (~20 FPS)
    elapsed = time.time() - start
    sleep_time = max(0, 0.05 - elapsed)
    time.sleep(sleep_time)

cap.release()
cv2.destroyAllWindows()
client.close()