from server.utils.drawing import draw_boxes


def run(frame, search_model):
    search_model.submit(frame)
    boxes = search_model.get_boxes()

    annotated = frame.copy()
    annotated = draw_boxes(annotated, boxes)

    return annotated