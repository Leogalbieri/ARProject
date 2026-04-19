import threading
from queue import Queue, Empty
from ultralytics import YOLO
from server import config

class SecondaryModel:
    def __init__(self):
        print("Loading Secondary Model...")
        self.model = YOLO(config.SECONDARY_MODEL)
        print("Secondary Model loaded!")
        self.queue = Queue(maxsize=1)
        self.boxes = []
        self.lock = threading.Lock()

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def submit(self, frame):
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Empty:
                break
        self.queue.put(frame)

    def get_boxes(self):
        with self.lock:
            return list(self.boxes)

    def _run(self):
        skip = 0

        while True:
            try:
                frame = self.queue.get(timeout=1)
            except Empty:
                continue

            skip += 1
            if skip % 3 != 0:
                continue

            results = self.model(
                frame,
                imgsz=config.IMG_SIZE,
                conf=config.CONF_THRESHOLD,
                device=config.DEVICE,
                verbose=False,
                half=True
            )

            new_boxes = []

            for box in results[0].boxes:
                label = self.model.names[int(box.cls[0])]

                if label in config.IGNORED_SECONDARY:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                new_boxes.append((x1, y1, x2, y2, label, conf))

            with self.lock:
                self.boxes = new_boxes