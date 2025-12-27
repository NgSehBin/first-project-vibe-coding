import os
import requests  # standard library, no install neededimport streamlit as st
import numpy as np
import torch
import cv2
from PIL import Image
from segment_anything import sam_model_registry, SamPredictor
from streamlit_image_coordinates import streamlit_image_coordinates
import io

# --- CONFIGURATION ---
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
CHECKPOINT_PATH = "sam_vit_b_01ec64.pth"
MODEL_TYPE = "vit_b"

def load_model():
    """
    Loads the SAM model. 
    Auto-downloads the model weights if they don't exist locally.
    """
    # 1. Check if model file exists, if not, download it
    if not os.path.exists(CHECKPOINT_PATH):
        st.info("Downloading AI Model (approx 375MB)... this may take a minute.")
        response = requests.get(CHECKPOINT_URL, stream=True)
        with open(CHECKPOINT_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success("Download complete!")

    # 2. Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    return predictor

# --- CORE LOGIC ---
def process_image(image_pil):
    """Converts PIL image to Numpy array for the AI model."""
    return np.array(image_pil.convert("RGB"))

def remove_background(image_np, mask):
    """Applies the mask to create a transparent background."""
    # Create an RGBA image (Red, Green, Blue, Alpha)
    h, w, _ = image_np.shape
    
    # Create the alpha channel based on the mask
    # Mask is True (object) or False (background)
    alpha_channel = np.zeros((h, w), dtype=np.uint8)
    alpha_channel[mask] = 255  # 255 is fully opaque
    
    # Split the original image
    b, g, r = cv2.split(image_np)
    
    # Merge them back with the new alpha channel
    rgba = [b, g, r, alpha_channel]
    masked_image = cv2.merge(rgba, 4)
    
    return masked_image

# --- UI LAYOUT ---
st.title("✂️ AI Smart Crop & Background Remover")
st.markdown("### Upload an image, click the subject, and download the result.")

# 1. Load the Model
try:
    with st.spinner("Loading AI Model... (This happens once)"):
        predictor = load_model()
except FileNotFoundError:
    st.error(f"Model file not found! Please download 'sam_vit_b_01ec64.pth' and place it in the project folder.")
    st.stop()

# 2. File Uploader
uploaded_file = st.file_uploader("Upload Image (PNG, JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load and prepare image
    image_pil = Image.open(uploaded_file)
    image_np = process_image(image_pil)
    
    # Set the image to the predictor
    predictor.set_image(image_np)

    col1, col2 = st.columns(2)

    with col1:
        st.header("1. Click on the Object")
        st.info("Click a point on the image to select the subject to keep.")
        
        # This component captures the X, Y coordinates of the click
        # We use a unique key to ensure it updates correctly
        value = streamlit_image_coordinates(image_pil, key="pil")

    with col2:
        st.header("2. Preview & Download")
        
        if value:
            # Get coordinates from the click
            input_point = np.array([[value["x"], value["y"]]])
            input_label = np.array([1])  # 1 indicates a foreground point

            with st.spinner("AI is removing the background..."):
                # Predict the mask
                masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=False,
                )
                
                # The model returns the best mask at index 0
                best_mask = masks[0]
                
                # Process the result
                result_image_cv2 = remove_background(image_np, best_mask)
                
                # Convert back to PIL for display/download
                result_image_pil = Image.fromarray(result_image_cv2)
                
                # Show Preview
                st.image(result_image_pil, caption="Cropped Output", use_container_width=True)
                
                # Convert to bytes for download
                buf = io.BytesIO()
                result_image_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()

                # Download Button
                st.download_button(
                    label="⬇️ Download Transparent PNG",
                    data=byte_im,
                    file_name="cropped_subject.png",
                    mime="image/png"
                )
        else:
            st.warning("Waiting for you to click on the image...")
