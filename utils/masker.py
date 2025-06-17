import numpy as np
from PIL import Image
import rasterio
import cv2

def calculate_ndwi(green_band, nir_band, epsilon=1e-6):
    """Calculate Normalized Difference Water Index"""
    return (green_band.astype(float) - nir_band.astype(float)) / (green_band + nir_band + epsilon)

def create_water_mask(image_path, ndwi_threshold=0.2):
    """
    Create water mask from image with enhanced water detection
    Returns binary mask (0=land, 255=water)
    """
    try:
        # Handle multi-band GeoTIFFs
        if image_path.lower().endswith(('.tif', '.tiff')):
            with rasterio.open(image_path) as src:
                # Try to get green and NIR bands
                bands = src.count
                if bands >= 4:  # Assume RGBN (Red, Green, Blue, NIR)
                    green = src.read(2)
                    nir = src.read(4)
                    ndwi = calculate_ndwi(green, nir)
                    mask = ((ndwi > ndwi_threshold) * 255).astype(np.uint8)
                else:  # Fallback to RGB
                    rgb = np.dstack([src.read(i) for i in range(1, min(4, bands)+1)])
                    blue = rgb[:,:,2].astype(float)
                    green = rgb[:,:,1].astype(float)
                    water_ratio = blue / (green + 1e-6)
                    mask = ((water_ratio > 1.1) * 255).astype(np.uint8)
        
        # Handle regular RGB images
        else:
            img = np.array(Image.open(image_path))
            if len(img.shape) == 3:
                blue = img[:,:,2].astype(float)
                green = img[:,:,1].astype(float)
                water_ratio = blue / (green + 1e-6)
                mask = ((water_ratio > 1.1) * 255).astype(np.uint8)
            else:  # Grayscale
                mask = np.zeros_like(img)
        
        # Post-processing to clean up small noise
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return mask
        
    except Exception as e:
        print(f"Error creating mask for {image_path}: {str(e)}")
        return np.zeros((256, 256), dtype=np.uint8)  # Return blank mask on error