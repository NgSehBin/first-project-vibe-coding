import streamlit as st
import numpy as np
from PIL import Image
import io
import requests
import os
import torch
import gc # Garbage Collection

# --- LIBRARIES FOR MODES ---
from rembg import remove # The new "Auto" library
from mobile_sam import sam_model_registry, SamPredictor
from streamlit_image_coordinates import streamlit_image_coordinates

# --- CONFIGURATION ---
st.set_page_config(page_title="Pro Background Remover", layout="centered")

# --- CACHING THE MANUAL MODEL (MobileSAM) ---
# We keep this for the manual tab, but only load it if needed
CHECKPOINT_URL = "https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt?raw=true"
CHECKPOINT_PATH = "mobile_sam.pt"

@st.cache_resource
def load_manual_model():
    """Loads the MobileSAM model only when Manual Mode is used."""
    if not os.path.exists(CHECKPOINT_PATH):
        with st.spinner("Downloading Manual Model..."):
            response = requests.get(CHECKPOINT_URL)
            with open(CHECKPOINT_PATH, "wb") as f:
                f.write(response.content)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_t"](checkpoint=CHECKPOINT_PATH)
    sam.to(device=device)
    sam.eval()
    return SamPredictor(sam)

# --- HELPER FUNCTIONS ---
def convert_image(image_pil):
    """Converts PIL to Bytes for download."""
    buf = io.BytesIO()
    image_pil.save(buf, format="PNG")
    return buf.getvalue()

def resize_image(image_pil, max_size=1024):
    """Resizes image to prevent memory crashes."""
    if max(image_pil.size) > max_size:
        image_pil.thumbnail((max_size, max_size), Image.LANCZOS)
    return image_pil

# --- APP UI ---
st.title("✂️ Pro Background Remover")

# Create Tabs
tab1, tab2 = st.tabs(["🤖 Auto Mode (Fast)", "👆 Manual Click (Precise)"])

# --- TAB 1: AUTO MODE (REMBG) ---
with tab1:
    st.header("One-Click Auto Removal")
    st.info("Best for: People, Products, and Cars. Detects the whole object automatically.")
    
    auto_file = st.file_uploader("Upload Image for Auto-Remove", type=["png", "jpg", "jpeg"], key="auto")
    
    if auto_file:
        image = Image.open(auto_file)
        image = resize_image(image)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original", use_container_width=True)
            
        # The Magic Button
        if st.button("🚀 Remove Background", type="primary"):
            with st.spinner("AI is detecting the subject..."):
                try:
                    # REMBG handles the heavy lifting here
                    output_image = remove(image)
                    
                    with col2:
                        st.image(output_image, caption="Result", use_container_width=True)
                        
                        # Download
                        st.download_button(
                            label="⬇️ Download Result",
                            data=convert_image(output_image),
                            file_name="auto_removed.png",
                            mime="image/png"
                        )
                except Exception as e:
                    st.error(f"Error: {e}")

# --- TAB 2: MANUAL MODE (MobileSAM) ---
with tab2:
    st.header("Manual Click Selection")
    st.info("Best for: Selecting a specific item in a group (e.g., one apple in a basket).")
    
    manual_file = st.file_uploader("Upload Image for Manual Click", type=["png", "jpg", "jpeg"], key="manual")

    if manual_file:
        # Load Model only now to save RAM
        predictor = load_manual_model()
        
        raw_image = Image.open(manual_file)
        raw_image = resize_image(raw_image, max_size=800)
        image_np = np.array(raw_image.convert("RGB"))
        
        # Prepare Predictor
        predictor.set_image(image_np)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Click the object:**")
            value = streamlit_image_coordinates(raw_image, key="pil_manual")
            
        with col2:
            if value:
                # Process Click
                input_point = np.array([[value["x"], value["y"]]])
                input_label = np.array([1])
                
                masks, _, _ = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=False
                )
                
                # Apply Mask
                mask = masks[0]
                h, w, _ = image_np.shape
                alpha = np.zeros((h, w), dtype=np.uint8)
                alpha[mask] = 255
                b, g, r = cv2.split(image_np) # Use OpenCV usually, but simple merge works too
                import cv2 # Ensure cv2 is imported inside scope if needed
                rgba = [b, g, r, alpha]
                result_cv2 = cv2.merge(rgba, 4)
                
                result_pil = Image.fromarray(result_cv2)
                st.image(result_pil, caption="Selected Crop", use_container_width=True)
                
                st.download_button(
                    label="⬇️ Download Crop",
                    data=convert_image(result_pil),
                    file_name="manual_crop.png",
                    mime="image/png"
                )
