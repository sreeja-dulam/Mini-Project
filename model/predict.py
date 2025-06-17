import numpy as np
from PIL import Image
from utils.masker import create_water_mask  # Import the create_water_mask function
import io
import tempfile
import matplotlib.pyplot as plt


def preprocess_image(uploaded_image):
    """
    Preprocess the uploaded image (e.g., resize, normalization, etc.)
    """
    img = Image.open(uploaded_image).convert('RGB')  # Open the image directly from the uploaded file
    img = img.resize((256, 256))  # Resize to model's expected input size
    img_array = np.array(img) / 255.0  # Normalize the image
    return img_array

def save_uploaded_file(uploaded_file):
    """Save the uploaded file temporarily and return the file path"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.tif')
    with open(temp_file.name, 'wb') as f:
        f.write(uploaded_file.getvalue())
    return temp_file.name

def predict_water_body(uploaded_image):
    """
    Predict the water body in the uploaded image using the water mask.
    This will also calculate the percentage of water in the image.
    """
    try:
        # Save the uploaded file temporarily
        temp_file_path = save_uploaded_file(uploaded_image)

        # Generate the water mask using the masker.py function
        water_mask = create_water_mask(temp_file_path)

        # Visualize the water mask to ensure it's working correctly
        plt.imshow(water_mask, cmap='gray')
        plt.title('Water Mask')
        plt.show()

        # Calculate the percentage of water (white pixels in the mask)
        white_pixels = np.sum(water_mask == 255)
        total_pixels = water_mask.size
        water_percentage = (white_pixels / total_pixels) * 100

        # Check if any water was detected (white pixels)
        print(f"White Pixels: {white_pixels}")
        print(f"Total Pixels: {total_pixels}")
        print(f"Water Percentage: {water_percentage}%")

        return water_mask, water_percentage

    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, 0.0

# Example usage:
# prediction_mask, percentage = predict_water_body(uploaded_image)
