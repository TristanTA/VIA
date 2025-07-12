import base64
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL_NAME = "gpt-4o-mini"

def image_to_base64(image_path):
    """Encodes an image file to base64."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")
    
def parse_label_and_confidence(result):
    """Extracts label and confidence from LLM reply."""
    label = ""
    confidence = 0
    for line in result.split("\n"):
        if line.lower().startswith("label:"):
            label = line.partition(":")[2].strip()
        elif line.lower().startswith("confidence:"):
            try:
                confidence = int(line.partition(":")[2].strip())
            except ValueError:
                confidence = 0
    return label, confidence

def query_llm_for_label(image_path):
    """Sends an image to GPT-4o Mini and retrieves label in structured format."""
    client = OpenAI()
    crop_b64 = image_to_base64(image_path)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert image analyst for a visual intelligence system. "
                    "You will be provided with a cropped image of an object from a snapshot of the user's view. "
                    "Provide a detailed but concise label for the object in the cropped image and confidence interval between 0 and 100, 100 being comletely confident and 0 being unknown. \n\n"
                    "Be as specfic as possible, providing object specific information that is useful for identification.\n\n"
                    "Good examples of labels include:\n 2012 Chevrolet Trailblazer, Apple iPhone 12 Pro Max, Dell XPS 13 Laptop\n\n"
                    "Bad examples of labels include:\n White Chevrolet, Apple iPhone, Laptop\n\n"
                    "Identify the object in the image and provide the response in the following format:\n\n"
                    "Label: 2012 Chevrolet Trailblazer\n"
                    "Confidence: 92\n"
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Identify this object using the cropped image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"}}
                ]
            }
        ],
        max_tokens=150,
    )

    result = response.choices[0].message.content.strip()
    label, confidence = parse_label_and_confidence(result)
    return label, confidence

def query_llm_for_label_with_context(full_image_path, crop_image_path):
    """Sends both full image and crop to GPT-4o Mini for labeling (fallback)."""
    client = OpenAI()
    full_b64 = image_to_base64(full_image_path)
    crop_b64 = image_to_base64(crop_image_path)
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert image analyst for a visual intelligence system. "
                    "You will be provided with a cropped image of an object from a snapshot of the user's view and an image of the entire scene. "
                    "Provide a detailed but concise label for the object in the cropped image and confidence interval between 0 and 100, 100 being comletely confident and 0 being unknown. \n\n"
                    "Be as specfic as possible, providing object specific information that is useful for identification.\n\n"
                    "Good examples of labels include:\n 2012 Chevrolet Trailblazer, Apple iPhone 12 Pro Max, Dell XPS 13 Laptop\n\n"
                    "Bad examples of labels include:\n White Chevrolet, Apple iPhone, Laptop\n\n"
                    "Identify the object in the image and provide the response in the following format:\n\n"
                    "Label: 2012 Chevrolet Trailblazer\n"
                    "Confidence: 92\n"
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Identify this object using the cropped image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{full_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"}}
                ]
            }
        ],
        max_tokens=150,
    )

    result = response.choices[0].message.content.strip()
    label, confidence = parse_label_and_confidence(result)
    return label, confidence
