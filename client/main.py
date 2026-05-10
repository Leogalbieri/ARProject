import multiprocessing
import subprocess
import sys


def run_audio_client():
    subprocess.run([sys.executable, "audio_client.py"])


def run_video_client():
    subprocess.run([sys.executable, "video_client.py"])


if __name__ == "__main__":
    audio_process = multiprocessing.Process(target=run_audio_client, name="AudioClient")
    video_process = multiprocessing.Process(target=run_video_client, name="VideoClient")

    audio_process.start()
    video_process.start()

    print("Both clients running...")
    print(f"Audio client PID: {audio_process.pid}")
    print(f"Video client PID: {video_process.pid}")

    try:
        audio_process.join()
        video_process.join()
    except KeyboardInterrupt:
        print("\nShutting down clients...")
        audio_process.terminate()
        video_process.terminate()
        audio_process.join()
        video_process.join()
        print("Done.")
