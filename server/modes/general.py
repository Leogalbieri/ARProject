from server.utils.drawing import draw_boxes

def run(frame, primary_model, secondary_model):
    annotated, primary_boxes = primary_model.infer(frame)

    secondary_model.submit(frame)
    secondary_boxes = secondary_model.get_boxes()

    annotated = draw_boxes(annotated, secondary_boxes)

    return annotated