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

def query_llm_for_information(image_path, label):
    """Sends an image to GPT-4o Mini and retrieves relevant info in structured format."""
    client = OpenAI()
    crop_b64 = image_to_base64(image_path)
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert image analyst for a visual intelligence system. "
                    "You will receive a cropped image of an object within that scene and the label for that object."
                    "Your task is to provide relevant information about the object that can be useful for identification or practical applications.\n\n"
                    "Relevant Info should be short phrases that are more application-oriented and helpful than just descriptive.\n"
                    "Provide only the relevant info, no additional text.\n\n"
                    "It is better to provide no information than to provide vague or generic or irrelvant information.\n\n"
                    "Relevant means it is useful for the user in some way, such as practical facts, application-oriented details, or specific characteristics that aid in identification.\n\n"
                    "Good examples of relevant info include:\n This fruit has a calorie count of roughly 90 kcal, This resturant closes at 10 pm (15 minutes from now), That dog is a Labrador Retriever \n\n"
                    "Bad examples of relevant info include:\n This fruit is red, This resturant is a fast food place, That dog is brown, This car is a sedan\n\n"
                    "Provide the response in the following format:\n\n"
                    "Relevant Info: [brief fact or practical info. If none, write 'None']"
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Provide relevant information for the following object."},
                    {"type": "text", "text": "Label: {label}"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{crop_b64}"}}
                ]
            }
        ],
        max_tokens=150,
    )
    result = response.choices[0].message.content.strip()

    info = ""

    for line in result.split("\n"):
        if line.lower().startswith("relevant info:"):
            info = line.partition(":")[2].strip()

    return {
        "info": info
    }

