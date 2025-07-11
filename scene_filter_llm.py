import base64
import json
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_NAME = "gpt-4o-mini"

def filter_relevant_objects(objects, user_preferences=None):
    """
    Filters relevant objects using scene-level LLM reasoning.
    Optionally takes user_preferences (string).
    """
    # Prepare structured prompt
    objects_description = ""
    for idx, obj in enumerate(objects, 1):
        objects_description += f"{idx}. Label: {obj['label']}, Info: {obj['info']}\n"

    prompt = (
        "You are a helpful assistant analyzing detected objects in an image.\n\n"
        "Here is the list of detected objects:\n"
        f"{objects_description}\n"
    )

    if user_preferences:
        prompt += f"\nUser preferences: {user_preferences}\n"

    prompt += (
        "\nBased on the above, select only the relevant objects for the user's interests.\n"
        "Reply ONLY with the numbers of relevant objects, separated by commas (e.g., 1, 3, 5).\n"
        "If none are relevant, reply with 'None'."
    )

    # Send to LLM
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You analyze scenes and filter objects based on relevance."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
    )

    reply = response.choices[0].message.content.strip()
    return parse_filter_reply(reply, objects)


def parse_filter_reply(reply, objects):
    """Parses the LLM reply and returns filtered objects."""
    if reply.lower() == "none":
        return []

    try:
        indices = [int(num.strip()) for num in reply.split(",")]
        filtered = [objects[i - 1] for i in indices if 1 <= i <= len(objects)]
        return filtered
    except Exception as e:
        print("Error parsing LLM reply:", reply, e)
        return []  # Fallback: nothing filtered


