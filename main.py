import os
import zipfile
import streamlit as st
from utils.tiler import tile_image
from utils.masker import create_water_mask
from model.predict import predict_water_body
import shutil
from PIL import Image


# Function to process and tile images
def process_and_tile_images(input_folder, tile_size, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    tiled_image_paths = []

    for filename in os.listdir(input_folder):
        if filename.endswith(('.tif', '.tiff')):  # Process only TIFF files
            image_path = os.path.join(input_folder, filename)
            # Ensure each tiled image is saved in the correct output folder
            tiles = tile_image(image_path, tile_size, output_folder)
            tiled_image_paths.extend(tiles)

    return tiled_image_paths

# Function to create water masks for tiled images
def create_masks_for_tiles(tiled_image_paths, mask_folder):
    os.makedirs(mask_folder, exist_ok=True)
    mask_paths = []
    
    for tile_path in tiled_image_paths:
        mask = create_water_mask(tile_path)
        mask_name = os.path.splitext(os.path.basename(tile_path))[0] + "_mask.png"
        mask_path = os.path.join(mask_folder, mask_name)
        mask_paths.append(mask_path)
        Image.fromarray(mask).save(mask_path)
    
    return mask_paths

# Function to zip a folder
def zip_folder(folder_path, zip_name):
    shutil.make_archive(zip_name, 'zip', folder_path)

# Streamlit app
def main():
    st.title("Satellite Image Preprocessing & Water Body Prediction")

    st.header("Upload Folder of TIFF Images")
    uploaded_folder = st.file_uploader("Choose a folder", type="zip", accept_multiple_files=False)

    if uploaded_folder is not None:
        # Extract the uploaded folder
        with open(os.path.join("uploads", uploaded_folder.name), "wb") as f:
            f.write(uploaded_folder.getbuffer())
        
        # Unzip the folder if it's a zip file
        if uploaded_folder.name.endswith(".zip"):
            with zipfile.ZipFile(os.path.join("uploads", uploaded_folder.name), "r") as zip_ref:
                zip_ref.extractall("uploads/")

        # Process the images: Tile them and create masks
        tiled_folder = "preprocessed_dataset/tiled_images"  # Path for tiled images
        mask_folder = "preprocessed_dataset/masks"  # Path for masks
        tiled_image_paths = process_and_tile_images("uploads/", 256, tiled_folder)
        mask_paths = create_masks_for_tiles(tiled_image_paths, mask_folder)

        # Zip the results
        zip_folder(tiled_folder, "preprocessed_dataset/tiled_images")
        zip_folder(mask_folder, "preprocessed_dataset/masks")

        st.success("Images processed and masks created.")
        st.download_button("Download Tiled Images", "preprocessed_dataset/tiled_images.zip")
        st.download_button("Download Water Masks", "preprocessed_dataset/masks.zip")
    
    # Water body prediction section
    st.header("Water Body Prediction")
    uploaded_image = st.file_uploader("Upload a Single TIFF Image for Prediction", type="tiff")
    
    if uploaded_image is not None:
        # Prediction logic (Use your trained model to predict)
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        # Placeholder for model prediction
        # Here you would load the trained model and make predictions
        water_percentage = predict_water_body(uploaded_image)  # This function should use your model
        
        st.write(f"Water Body Percentage: {water_percentage}%")

# Main function
if __name__ == "__main__":
    main()
