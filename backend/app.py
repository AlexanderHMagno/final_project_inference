import gradio as gr
from inference import detect_people

# === Gradio Interface ===
demo = gr.Interface(
    fn=detect_people,
    inputs=gr.Image(type="pil", label="Upload Large Image"),
    outputs=[
        gr.Image(type="pil", label="Patch-by-Patch Detection"),
        gr.Image(type="pil", label="Final Detection Result")
    ],
    title="🧍 YOLOv11 Person Detector",
    description="Uploads a large image, splits it into patches, detects people in each patch, and shows both the patch analysis and final merged result."
)

# === Launch ===
if __name__ == "__main__":
    demo.launch()