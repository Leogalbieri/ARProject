from ultralytics import YOLO
from server import config

class PrimaryModel:
    def __init__(self):
        print("Loading Primary Model...")
        self.model = YOLO(config.PRIMARY_MODEL)
        print("Primary Model loaded!")

    def infer(self, frame):
        results = self.model(
            frame,
            imgsz=config.IMG_SIZE,
            conf=config.CONF_THRESHOLD,
            device=config.DEVICE,
            verbose=False,
            half=True
        )

        filtered = []
        for box in results[0].boxes:
            label = self.model.names[int(box.cls[0])]

            if label in config.IGNORED_PRIMARY:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            filtered.append((x1, y1, x2, y2, label, conf))

        return results[0].plot(), filtered