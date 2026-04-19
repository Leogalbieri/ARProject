import cv2

def draw_boxes(frame, boxes, color=(255, 100, 0)):
    for (x1, y1, x2, y2, label, conf) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame