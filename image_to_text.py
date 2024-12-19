import streamlit as st
import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from huggingface_hub import login
import io

# Authenticate with Hugging Face
HF_TOKEN = st.secrets["newfinegrained"]
login(HF_TOKEN)

def load_model_and_processor(model_id):
    """Load the model and processor."""
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

def generate_text(model, processor, image_url, prompt):
    """Generate text using the model and processor."""
    try:
        # Fetch the image from the URL
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an error for invalid response
        
        # Validate content type
        if "image" not in response.headers["Content-Type"]:
            return "Error: The provided URL does not point to a valid image."

        # Open the image
        image = Image.open(io.BytesIO(response.content))

        # Process the image and prompt
        inputs = processor(image, prompt, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=30)

        # Decode the output
        return processor.decode(output[0])
    except Exception as e:
        return f"Error: {e}"


# Streamlit App
st.title("LLaMA 3.2 Vision")

# Model ID and loading
MODEL_ID = "meta-llama/Llama-3.2-11B-Vision"
model, processor = load_model_and_processor(MODEL_ID)

# User input for image URL and prompt
image_url = st.text_input(
    "Enter the Image URL:",
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
)
prompt = st.text_area(
    "Enter your prompt:",
    "<|image|><|begin_of_text|>If I had to write a haiku for this one"
)

# Button to generate haiku
if st.button("Generate Text"):
    with st.spinner("Generating Text..."):
        result = generate_text(model, processor, image_url, prompt)
    
    st.subheader("Generated Text")
    st.write(result)

    try:
        st.image(image_url, caption="Input Image")
    except Exception:
        st.error("Failed to load image. Please check the URL.")