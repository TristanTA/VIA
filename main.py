from models.yolov8_model import load_model, detect_objects
from utils.image_utils import draw_boxes

image_path = 'data/sample.jpg'
output_path = 'outputs/output.jpg'

model = load_model()
results = detect_objects(model, image_path)
draw_boxes(results, image_path, output_path)

print(f"Detection complete! Output saved to {output_path}")