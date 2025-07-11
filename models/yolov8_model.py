from ultralytics import YOLO

def load_model(model_path='yolov8n.pt'):
    model = YOLO(model_path)
    return model

def detect_objects(model, image_path):
    results = model(image_path, conf=0.40)
    return results