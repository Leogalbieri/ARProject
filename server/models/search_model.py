import threading
from queue import Queue, Empty
from ultralytics import YOLO
from server import config

class SearchModel:
    def __init__(self):
        print("Loading search model...")
        self.model = YOLO(config.SEARCH_MODEL)
        print("Search model loaded!")
        self.targets = config.SEARCH_TARGET
        self.model.set_classes(self.targets)
        self.queue = Queue(maxsize=1)
        self.boxes = []
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def set_targets(self, targets):
        self.targets = targets
        self.model.set_classes(targets)

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
                conf=config.CONF_THRESHOLD_SEARCH,
                device=config.DEVICE,
                verbose=False,
                half=True
            )

            new_boxes = []
            for box in results[0].boxes:
                label = self.model.names[int(box.cls[0])]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                new_boxes.append((x1, y1, x2, y2, label, conf))

            with self.lock:
                self.boxes = new_boxes