import cv2
import os
import shutil

CROPS_DIR = "crops"
os.makedirs(CROPS_DIR, exist_ok=True)

def extract_detections(results):
    """Extracts bounding boxes from YOLO results."""
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append([x1, y1, x2, y2])
    return detections

def crop_and_save(image_path, detections):
    """Crops bounding boxes and saves them as individual images."""
    image = cv2.imread(image_path)
    crops = []
    for idx, bbox in enumerate(detections):
        xmin, ymin, xmax, ymax = bbox
        crop = image[ymin:ymax, xmin:xmax]
        crop_path = os.path.join(CROPS_DIR, f"crop_{idx}.jpg")
        cv2.imwrite(crop_path, crop)
        crops.append({
            "crop_path": crop_path,
            "bbox": bbox
        })
    return crops

def clear_folder(folder_path):
    """Deletes all files in the given folder."""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)
