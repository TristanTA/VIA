from models.yolov8_model import load_model, detect_objects
from utils.image_utils import draw_boxes
from image_processing import extract_detections, crop_and_save, clear_folder
from scene_filter_llm import filter_relevant_objects
from object_labeler_llm import query_llm_for_label, query_llm_for_label_with_context
from information_retrieval_llm import query_llm_for_information
from feedback_ui import show_feedback_ui
import json
from tqdm import tqdm
import os
import base64

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

# Step 4: Query LLM for each crop (labels only — skip info for now)
print("Querying LLM for labels...")
objects = []
for crop in tqdm(crops, desc="Labeling objects"):
    label, confidence = query_llm_for_label(crop["crop_path"])

    if not (0 <= confidence <= 100):
        confidence = 0

    if label == " ":
        print("Fallback also failed to produce a label.")
        label = "Unknown"
    
    if confidence < 60:
        print(f"Low confidence ({confidence}%) — using full scene context.")
        label, _ = query_llm_for_label_with_context(image_path, crop["crop_path"])
        print(f"Label after fallback: {label}")
    
    crop["label"] = label
    
    obj = {
        "crop_path": crop["crop_path"],
        "bbox": crop["bbox"],
        "label": crop["label"],
        "relevant_info": "None",  # Placeholder for now
    }
    objects.append(obj)

# Step 5: Filter relevant objects (only after labeling)
print("Filtering relevant objects...")
filtered_objects, relevance_score = filter_relevant_objects(objects, user_preferences="None")

print(f"Relevance score for scene: {relevance_score}")
print(f"Objects kept after filtering: {len(filtered_objects)}")

# Step 6: Retrieve relevant info only for kept objects
print("Querying LLM for detailed info on filtered objects...")
for obj in tqdm(filtered_objects, desc="Retrieving info"):
    info = query_llm_for_information(obj["crop_path"], obj["label"])
    if not info or "info" not in info:
        info = {"info": "None"}
    obj["relevant_info"] = info["info"]

# Step 7: Save feedback using the feedback UI
show_feedback_ui(output_path, filtered_objects)

# Remove cropped images of discarded objects
all_kept_paths = {obj["crop_path"] for obj in filtered_objects}
for obj in objects:
    if obj["crop_path"] not in all_kept_paths:
        os.remove(obj["crop_path"])

# Save results
with open(output_json_path, "w") as f:
    json.dump({
        "image_path": image_path,
        "objects": filtered_objects
    }, f, indent=4)

print(f"Workflow complete! Results saved to {output_json_path}")
