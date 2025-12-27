import os
import requests
import numpy as np
import streamlit as st
import torch
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from streamlit_image_coordinates import streamlit_image_coordinates
import io
import gc # Garbage collection to free up memory

# --- CONFIGURATION ---
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
CHECKPOINT_PATH = "sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Background Remover", layout="centered")

@st.cache_resource
def load_model():
    """
    Loads the SAM model. 
    Auto-downloads the model weights if they don't exist locally.
    """
    # Create a placeholder for status messages
    status_msg = st.empty()
    
    # 1. Check if model file exists, if not, download it
    if not os.path.exists(CHECKPOINT_PATH):
        status_msg.info("Downloading AI Model (approx 375MB)... this happens only once.")
        try:
            response = requests.get(CHECKPOINT_URL, stream=True)
            with open(CHECKPOINT_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            # Clear the message after success
            status_msg.empty()
        except Exception as e:
            status_msg.error(f"Download failed: {e}")
            st.stop()

    # 2. Load the model
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        return predictor
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        st.stop()

# --- CORE LOGIC ---
def process_image(image_pil):
    """
    Converts PIL image to Numpy array for the AI model.
    CRITICAL FIX: Resizes large images to prevent RAM crashes.
    """
    # Resize if the image is too large (max 1024px on long side)
    max_dimension = 1024
    if max(image_pil.size) > max_dimension:
        image_pil.thumbnail((max_dimension, max_dimension), Image.LANCZOS)
    
    return np.array(image_pil.convert("RGB")), image_pil

def remove_background(image_np, mask):
    """Applies the mask to create a transparent background."""
    h, w, _ = image_np.shape
    
    # Create the alpha channel (0=transparent, 255=opaque)
    alpha_channel = np.zeros((h, w), dtype=np.uint8)
    alpha_channel[mask] = 255
    
    # Split the original image
    b, g, r = cv2.split(image_np)
    
    # Merge them back with the new alpha channel
    rgba = [b, g, r, alpha_channel]
    masked_image = cv2.merge(rgba, 4)
    
    return masked_image

# --- UI LAYOUT ---
st.title("✂️ AI Smart Crop")
st.markdown("Upload an image, click the object, and download without background.")

# 1. Load the Model (with hidden notification if successful)
predictor = load_model()

# 2. File Uploader
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and prepare image
    raw_image = Image.open(uploaded_file)
    
    # Process (Resize + Convert)
    image_np, image_pil = process_image(raw_image)
    
    # Force garbage collection to prevent memory leaks
    gc.collect()
    
    # Set the image to the predictor
    # We use a spinner here because this is the heavy step that was crashing before
    with st.spinner("Processing image..."):
        predictor.set_image(image_np)

    st.write("---")
    
    # UI Columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Click Object")
        # Display image and get click coordinates
        # value returns {'x': 123, 'y': 456}
        value = streamlit_image_coordinates(image_pil, key="pil")

    with col2:
        st.subheader("2. Result")
        
        if value:
            # Get coordinates from the click
            input_point = np.array([[value["x"], value["y"]]])
            input_label = np.array([1])  # 1 = Foreground

            # Run the AI
            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )
            
            # Use the best mask
            best_mask = masks[0]
            
            # Remove Background
            result_image_cv2 = remove_background(image_np, best_mask)
            result_image_pil = Image.fromarray(result_image_cv2)
            
            # Display Result
            st.image(result_image_pil, caption="Result", use_container_width=True)
            
            # Prepare Download
            buf = io.BytesIO()
            result_image_pil.save(buf, format="PNG")
            byte_im = buf.getvalue()

            st.download_button(
                label="⬇️ Download PNG",
                data=byte_im,
                file_name="transparent_subject.png",
                mime="image/png"
            )
        else:
            st.info("👆 Click the image on the left to start.")
