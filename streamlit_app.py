import os
import requests
import numpy as np
import streamlit as st
import torch
import cv2
from PIL import Image
# We now import from mobile_sam instead of segment_anything
from mobile_sam import sam_model_registry, SamPredictor
from streamlit_image_coordinates import streamlit_image_coordinates
import io
import gc

# --- CONFIGURATION ---
# We use the MobileSAM model (40MB) instead of the heavy one (375MB)
CHECKPOINT_URL = "https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt?raw=true"
CHECKPOINT_PATH = "mobile_sam.pt"
MODEL_TYPE = "vit_t"

st.set_page_config(page_title="Lightweight Background Remover", layout="centered")

@st.cache_resource
def load_model():
    """Loads the MobileSAM model. Small, fast, and crash-proof."""
    status_msg = st.empty()
    
    if not os.path.exists(CHECKPOINT_PATH):
        status_msg.info("Downloading Lightweight AI (40MB)...")
        try:
            response = requests.get(CHECKPOINT_URL)
            with open(CHECKPOINT_PATH, "wb") as f:
                f.write(response.content)
            status_msg.empty()
        except Exception as e:
            st.error(f"Download failed: {e}")
            st.stop()

    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # MobileSAM uses 'vit_t'
        sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
        sam.to(device=device)
        sam.eval() # Force evaluation mode for speed
        predictor = SamPredictor(sam)
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

def process_image(image_pil):
    """Resizes huge images to safe limits (800px) to protect RAM."""
    max_dimension = 800
    if max(image_pil.size) > max_dimension:
        image_pil.thumbnail((max_dimension, max_dimension), Image.LANCZOS)
    
    return np.array(image_pil.convert("RGB")), image_pil

def remove_background(image_np, mask):
    """Applies the mask to create transparency."""
    h, w, _ = image_np.shape
    alpha_channel = np.zeros((h, w), dtype=np.uint8)
    alpha_channel[mask] = 255
    b, g, r = cv2.split(image_np)
    rgba = [b, g, r, alpha_channel]
    return cv2.merge(rgba, 4)

# --- UI START ---
st.title("⚡ Fast Background Remover")
st.markdown("Upload image -> Click object -> Download.")

# 1. Load Model (Cached)
predictor = load_model()

# 2. Upload
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Prepare image
    raw_image = Image.open(uploaded_file)
    image_np, image_pil = process_image(raw_image)
    
    # Clean memory
    gc.collect()

    # Embed image (The heavy part)
    with st.spinner("Analyzing image..."):
        predictor.set_image(image_np)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Click Object")
        # Capture click
        value = streamlit_image_coordinates(image_pil, key="pil")

    with col2:
        st.subheader("2. Result")
        if value:
            # Get coordinates
            input_point = np.array([[value["x"], value["y"]]])
            input_label = np.array([1])

            # Predict
            masks, _, _ = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )
            
            # Create Output
            result_image = remove_background(image_np, masks[0])
            result_pil = Image.fromarray(result_image)
            
            st.image(result_pil, use_container_width=True)
            
            # Download
            buf = io.BytesIO()
            result_pil.save(buf, format="PNG")
            st.download_button(
                label="⬇️ Download PNG",
                data=buf.getvalue(),
                file_name="cutout.png",
                mime="image/png"
            )
        else:
            st.info("👆 Click the left image to start.")
