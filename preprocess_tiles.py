import os
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils.tiler import tile_image
from utils.masker import create_water_mask

def process_single_image(input_path, output_image_dir, output_mask_dir, tile_size=256):
    """Process one image into multiple tiles with corresponding masks"""
    try:
        # Create temp directory for tiles
        temp_dir = "temp_tiles"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate all tiles for this image
        tile_paths = tile_image(input_path, tile_size, temp_dir)
        
        # Process each tile
        for tile_path in tqdm(tile_paths, desc=f"Processing {os.path.basename(input_path)}"):
            # Generate water mask
            mask = create_water_mask(tile_path)
            
            # Save to final directories
            base_name = os.path.basename(tile_path)
            final_image_path = os.path.join(output_image_dir, base_name)
            final_mask_path = os.path.join(output_mask_dir, base_name)
            
            os.rename(tile_path, final_image_path)
            Image.fromarray(mask).save(final_mask_path)
            
        shutil.rmtree(temp_dir)
        return len(tile_paths)
        
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return 0

def preprocess_dataset(input_dir, output_image_dir, output_mask_dir, tile_size=256):
    """Process all images in input directory"""
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg'))]
    total_tiles = 0
    
    for img_file in tqdm(image_files, desc="Processing dataset"):
        input_path = os.path.join(input_dir, img_file)
        tiles_created = process_single_image(input_path, output_image_dir, output_mask_dir, tile_size)
        total_tiles += tiles_created
    
    print(f"\nPreprocessing complete! Created {total_tiles} tiles and masks.")

if __name__ == '__main__':
    config = {
        'input_dir': 'raw_satellite_images',
        'output_image_dir': 'dataset/images_tiled',
        'output_mask_dir': 'dataset/masks_tiled',
        'tile_size': 256
    }
    preprocess_dataset(**config)