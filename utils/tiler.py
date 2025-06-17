import os
import numpy as np
from PIL import Image
import rasterio
from rasterio.windows import Window

def tile_image(image_path, tile_size, output_dir):
    """Splits an image into multiple tiles. Returns paths to all tiles."""
    os.makedirs(output_dir, exist_ok=True)
    tile_paths = []
    
    if image_path.endswith(('.tif', '.tiff')):
        with rasterio.open(image_path) as src:
            for y in range(0, src.height, tile_size):
                for x in range(0, src.width, tile_size):
                    window = Window(x, y, tile_size, tile_size)
                    tile = src.read(window=window)
                    tile = np.moveaxis(tile, 0, -1)  # (C,H,W) → (H,W,C)
                    tile_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_{y}_{x}.png"
                    tile_path = os.path.join(output_dir, tile_name)
                    Image.fromarray(tile).save(tile_path)
                    tile_paths.append(tile_path)
    else:
        img = Image.open(image_path)
        for y in range(0, img.height, tile_size):
            for x in range(0, img.width, tile_size):
                tile = img.crop((x, y, x+tile_size, y+tile_size))
                tile_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_{y}_{x}.png"
                tile_path = os.path.join(output_dir, tile_name)
                tile.save(tile_path)
                tile_paths.append(tile_path)
    
    return tile_paths  # Returns list of ALL generated tiles