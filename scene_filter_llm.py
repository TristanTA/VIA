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
        objects_description += f"{idx}. Label: {obj['label']}, Info: {obj['relevant_info']}\n"

    prompt = (
    "You are a helpful assistant analyzing detected objects in an image.\n\n"
    f"Objects detected:\n{objects_description}\n"
)

    if user_preferences:
        prompt += f"\nUser preferences (if provided): {user_preferences}\n"

    prompt += (
        "\nSelect only the relevant objects based on user interests.\n"
        "- Reply with the numbers of relevant objects (e.g., 1, 3, 5) and a relevance score between 0 and 1.\n"
        "- The relvance score is used to train the model, so it is important to provide a score even if no objects are selected.\n"
        "You analyze scenes and filter objects based on relevance. Your responses are used to improve future results through feedback."
        "- Relevance means it is useful for the user in some way, such as practical facts, application-oriented details, or specific characteristics that aid in identification.\n"
        "- If none are relevant, reply 'None' with a relevance score (still required for training).\n"
        "- Include only useful objects (practical info, identification details, etc.).\n"
        "- Irrelevant objects must be excluded, even if unsure.\n\n"
        "Reply format:\n"
        "<indices>; <relevance_score>\n"
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
    """Parses the LLM reply and returns filtered objects and relevance score."""
    try:
        indices_part, score_part = reply.strip().split(";")
        relevance_score = float(score_part.strip())

        if indices_part.strip().lower() == "none":
            return [], relevance_score

        indices = [int(num.strip()) for num in indices_part.strip().split(",")]
        filtered_objects = [objects[i - 1] for i in indices if 1 <= i <= len(objects)]

        return filtered_objects, relevance_score

    except Exception as e:
        print("Error parsing LLM reply:", reply, e)
        return [], 0.0 



