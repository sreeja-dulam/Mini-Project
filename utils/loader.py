import os
import numpy as np
from PIL import Image
from tensorflow.keras.utils import Sequence
import rasterio

class SatelliteDataLoader(Sequence):
    def __init__(self, image_dir, mask_dir, batch_size=8, tile_size=256, shuffle=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.tile_size = tile_size
        self.shuffle = shuffle
        
        # Verify directories exist
        if image_dir and mask_dir:
            if not os.path.exists(image_dir):
                raise FileNotFoundError(f"Image directory not found: {image_dir}")
            if not os.path.exists(mask_dir):
                raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
            
            self.image_paths = sorted([
                os.path.join(image_dir, f) for f in os.listdir(image_dir)
                if f.endswith(('.png', '.jpg', '.tif'))
            ])
            self.mask_paths = sorted([
                os.path.join(mask_dir, f) for f in os.listdir(mask_dir)
                if f.endswith(('.png', '.jpg', '.tif'))
            ])
            
            # Verify pairing
            assert len(self.image_paths) == len(self.mask_paths), "Mismatched image/mask counts"
            for img, msk in zip(self.image_paths, self.mask_paths):
                assert os.path.basename(img) == os.path.basename(msk), f"Mismatched pairs: {img} vs {msk}"
        else:
            self.image_paths = []
            self.mask_paths = []
        
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = (index + 1) * self.batch_size
        indices = range(start_idx, min(end_idx, len(self.image_paths)))
        
        batch_images = []
        batch_masks = []
        
        for i in indices:
            # Load image
            img = Image.open(self.image_paths[i])
            img = img.resize((self.tile_size, self.tile_size))
            img_array = np.array(img) / 255.0
            
            # Load mask
            mask = Image.open(self.mask_paths[i]).convert('L')
            mask = mask.resize((self.tile_size, self.tile_size))
            mask_array = (np.array(mask) > 128).astype(np.float32)
            
            batch_images.append(img_array)
            batch_masks.append(np.expand_dims(mask_array, -1))
        
        return np.array(batch_images), np.array(batch_masks)

    def on_epoch_end(self):
        if self.shuffle and len(self.image_paths) > 0:
            combined = list(zip(self.image_paths, self.mask_paths))
            np.random.shuffle(combined)
            self.image_paths, self.mask_paths = zip(*combined)

    def split(self, val_ratio=0.2):
        """Split the dataset into training and validation sets"""
        if len(self.image_paths) == 0:
            raise ValueError("No images found to split")
            
        split_idx = int(len(self.image_paths) * (1 - val_ratio))
        
        # Create new loader instances
        train_loader = SatelliteDataLoader(
            image_dir=self.image_dir,
            mask_dir=self.mask_dir,
            batch_size=self.batch_size,
            tile_size=self.tile_size,
            shuffle=self.shuffle
        )
        val_loader = SatelliteDataLoader(
            image_dir=self.image_dir,
            mask_dir=self.mask_dir,
            batch_size=self.batch_size,
            tile_size=self.tile_size,
            shuffle=False
        )
        
        # Manually set the paths
        train_loader.image_paths = self.image_paths[:split_idx]
        train_loader.mask_paths = self.mask_paths[:split_idx]
        val_loader.image_paths = self.image_paths[split_idx:]
        val_loader.mask_paths = self.mask_paths[split_idx:]
        
        return train_loader, val_loader