import multiprocessing
import subprocess
import sys


def run_audio_server():
    subprocess.run([sys.executable, "audio_server.py"])


def run_video_server():
    subprocess.run([sys.executable, "video_server.py"])


if __name__ == "__main__":
    audio_process = multiprocessing.Process(target=run_audio_server, name="AudioServer")
    video_process = multiprocessing.Process(target=run_video_server, name="VideoServer")

    audio_process.start()
    video_process.start()

    print("Both servers running...")
    print(f"Audio server PID: {audio_process.pid}")
    print(f"Video server PID: {video_process.pid}")

    try:
        audio_process.join()
        video_process.join()
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        audio_process.terminate()
        video_process.terminate()
        audio_process.join()
        video_process.join()
        print("Done.")
