# main.py

from models.yolov8_model import load_model, detect_objects
from utils.image_utils import draw_boxes
from image_processing import extract_detections, crop_and_save, clear_folder
from scene_filter_llm import filter_relevant_objects
from object_labeler_llm import query_llm_for_label
from information_retrieval_llm import query_llm_for_information
import json
from tqdm import tqdm
import os

image_path = 'data/sample.jpg'
output_path = 'outputs/output.jpg'
output_json_path = "outputs/filtered_labels.json"

print("Starting the object detection and labeling workflow...")
clear_folder("crops")

# Step 1: Detect objects
print("Loading model and detecting objects...")
yolo_model = load_model()
results = detect_objects(yolo_model, image_path)
print("Drawing bounding boxes on the image...")
draw_boxes(results, image_path, output_path)

# Step 2: Extract bounding boxes
print("Extracting bounding boxes from detection results...")
detections = extract_detections(results)

# Step 3: Crop detected regions
print("Cropping detected regions and saving them...")
crops = crop_and_save(image_path, detections)

# Step 4: Query LLM for each crop
print("Querying LLM for labels and relevant info...")
objects = []
for crop in tqdm(crops, desc="Labeling objects"):
    label = query_llm_for_label(crop["full_image_path"], crop["crop_path"])
    crop.update(label)
    info = query_llm_for_information(crop["full_image_path"], crop["crop_path"])
    crop.update(info)
    
    obj = {
        "crop_path": crop["crop_path"],
        "bbox": crop["bbox"],
        "label": crop["label"],
        "relevant_info": crop["info"],
    }
    objects.append(obj)

print("Objects labeled:", len(objects))
filtered_objects = filter_relevant_objects(objects, user_preferences="None")

kept_indices = [
    objects.index(obj) for obj in filtered_objects
]
for idx, obj in enumerate(objects):
    if idx not in kept_indices:
        os.remove(obj["crop_path"])
with open(output_json_path, "w") as f:
    json.dump({
        "image_path": image_path,
        "objects": filtered_objects  # Only relevant ones
    }, f, indent=4)

print(f"Workflow complete! Results saved to {output_json_path}")
