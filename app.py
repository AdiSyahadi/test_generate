import streamlit as st
import torch
from diffusers import AutoPipelineForText2Image, LCMScheduler
from diffusers.utils import make_image_grid, load_image
from PIL import Image
import os
import time
from io import BytesIO

def generate_image(init_image, prompt):
    # If using CPU
    pipe = AutoPipelineForImage2Image.from_pretrained(
        "Lykon/dreamshaper-7",
        variant="fp16"
    )

    # set scheduler
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    # load LCM-LoRA
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    pipe.fuse_lora()

    # pass prompt and image to pipeline
    generator = torch.manual_seed(0)
    image = pipe(
        prompt,
        image=init_image,
        num_inference_steps=4,
        guidance_scale=1,
        strength=0.6,
        generator=generator
    ).images[0]

    return image

def main():
    st.title("Image Generation with Streamlit")

    # Image source selection
    source = st.radio("Select Image Source", ("Local", "URL"))

    if source == "Local":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Convert BytesIO to PIL Image
            init_image = Image.open(BytesIO(uploaded_file.read()))
            st.image(init_image, caption="Uploaded Image", use_column_width=True)
    else:
        url = st.text_input("Enter Image URL:")
        if url:
            init_image = load_image(url)
            st.image(init_image, caption="Image from URL", use_column_width=True)

    prompt = st.text_input("Enter Instruction Prompt:", "Original face, self-portrait oil painting, a handsome cyborg with golden hair, 8k")

    if st.button("Generate Image"):
        # Mengukur waktu awal
        start_time = time.time()

        generated_image = generate_image(init_image, prompt)

        # Mengukur waktu akhir
        end_time = time.time()

        # Menghitung latensi dalam detik
        latency_seconds = end_time - start_time

        # Konversi latensi ke menit dan detik
        latency_minutes = int(latency_seconds // 60)
        remaining_seconds = latency_seconds % 60

        st.image(make_image_grid([init_image, generated_image], rows=1, cols=2), caption=f"Latency: {latency_minutes} minutes {remaining_seconds:.2f} seconds", use_column_width=True)

if __name__ == "__main__":
    main()
