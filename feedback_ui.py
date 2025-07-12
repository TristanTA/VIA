import gradio as gr
import json
from PIL import Image

def save_feedback(scores, objects, image_path):
    """Save feedback scores to JSON file."""
    feedback = []
    for idx, score in enumerate(scores):
        obj = objects[idx]
        feedback.append({
            "object_index": idx,
            "label": obj["label"],
            "bbox": obj["bbox"],
            "relevance_score": obj.get("relevance_score", "N/A"),
            "feedback_score": score,
        })
    with open("outputs/feedback.json", "w") as f:
        json.dump(feedback, f, indent=4)
    return "Feedback saved successfully!"

def show_feedback_ui(image_path, filtered_objects):
    """Display Gradio UI for object feedback."""
    image = Image.open(image_path)
    sliders = []
    with gr.Blocks() as demo:
        gr.Image(image, label="Scene with Detected Objects")
        with gr.Column():
            for obj in filtered_objects:
                slider = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.5,
                    step=0.01,
                    label=f"{obj['label']} (Relevance: {obj.get('relevance_score', 'N/A')})"
                )
                sliders.append(slider)
        submit = gr.Button("Save Feedback")
        result = gr.Textbox(label="Result")
        submit.click(
            fn=save_feedback,
            inputs=[*sliders, gr.State(filtered_objects), gr.State(image_path)],
            outputs=result
        )
    demo.launch()
