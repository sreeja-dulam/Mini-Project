import os
import zipfile
import streamlit as st
from utils.tiler import tile_image
from utils.masker import create_water_mask
from model.predict import predict_water_body
from io import BytesIO
from PIL import Image
import shutil

# Set custom page config
st.set_page_config(page_title="Satellite Image Preprocessing and Water Body Detection", layout="wide")

# Apply custom CSS for fonts and style
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Fredoka+One&family=Raleway:wght@700&family=Baloo+2&family=Caveat&display=swap');

        body {
            background-color: #000000;
            color: white;
        }
        .title {
            font-family: 'Fredoka One', cursive;
            color: white;
            font-size: 40px;
            text-align: center;
            margin-bottom: 10px;
        }
        .caption {
            font-family: 'Caveat', cursive;
            color: white;
            font-size: 35px;
            text-align: center;
            margin-top: 0px;
            margin-bottom: 40px;
        }
        .sidebar .sidebar-content {
            font-family: 'Baloo 2', cursive;
            font-size: 25px;
            color: white;
        }
        .stRadio > div {
            font-size: 23px;
            font-family: 'Baloo 2', cursive;
        }
        .stFileUploader label {
            font-size: 22px;
            font-family: 'Baloo 2', cursive;
        }
        .stButton button {
            font-size: 22px;
            font-family: 'Baloo 2', cursive;
        }
    </style>
""", unsafe_allow_html=True)

# Title and caption
st.markdown('<div class="title">🛰️ Satellite Image Preprocessing and Water Body Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="caption">🌍 .Tile . Mask . Detect. 🌊</div>', unsafe_allow_html=True)

# Sidebar option for upload
option = st.sidebar.radio("🚀Choose an Option", ["Upload Folder for Preprocessing", "Upload Single Image for Prediction"])

def handle_folder_upload(uploaded_folder):
    """Handle folder upload and preprocessing."""
    output_dir = "preprocessed_dataset"
    os.makedirs(output_dir, exist_ok=True)

    tiled_image_folder = os.path.join(output_dir, "tiled_images")
    mask_image_folder = os.path.join(output_dir, "masked_images")

    os.makedirs(tiled_image_folder, exist_ok=True)
    os.makedirs(mask_image_folder, exist_ok=True)
    
    # Unzip folder content
    with zipfile.ZipFile(uploaded_folder, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    
    # Process each image in the uploaded folder
    image_paths = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith(('.tif', '.tiff')):  # Process only TIFF files
                image_paths.append(os.path.join(root, file))
    
    # Tile the images and save them in the tiled_images folder
    tiled_image_paths = []
    for image_path in image_paths:
        tiles = tile_image(image_path, tile_size=256, output_dir=tiled_image_folder)
        tiled_image_paths.extend(tiles)
    
    # Generate masks for tiled images and save them in the masked_images folder
    mask_paths = []
    for image_path in tiled_image_paths:
        mask = create_water_mask(image_path)
        mask_name = os.path.splitext(os.path.basename(image_path))[0] + "_mask.png"
        mask_path = os.path.join(mask_image_folder, mask_name)
        mask_paths.append(mask_path)
        Image.fromarray(mask).save(mask_path)
    
    # Create a zip file of the preprocessed dataset folder
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), output_dir))

    zip_buffer.seek(0)
    return zip_buffer

def display_prediction(uploaded_image):
    """Show prediction result for water body percentage."""
    prediction, percentage = predict_water_body(uploaded_image)
    st.image(prediction, caption="Predicted Mask 🌊", use_column_width=True)
    st.write(f"Water Body Percentage Detected: {percentage}% 🛰️")

# Main functionality
if option == "Upload Folder for Preprocessing":
    uploaded_folder = st.file_uploader("📂 Upload Zipped Folder of Images", type="zip")
    if uploaded_folder:
        zip_buffer = handle_folder_upload(uploaded_folder)
        st.success("✅ Folder processed successfully! Ready to download ⬇️")
        st.download_button(
            label="⬇️ Download Preprocessed Dataset",
            data=zip_buffer,
            file_name="preprocessed_dataset.zip",
            mime="application/zip"
        )

elif option == "Upload Single Image for Prediction":
    uploaded_image = st.file_uploader("🖼️ Upload Single Satellite Image (.tiff)", type="tiff")
    if uploaded_image:
        display_prediction(uploaded_image)
