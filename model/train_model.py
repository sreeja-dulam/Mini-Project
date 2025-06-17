import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))
from utils.loader import SatelliteDataLoader
from unet import unet_model

def verify_paths(config):
    """Verify all required paths exist"""
    required_paths = [
        config['train_image_dir'],
        config['train_mask_dir'],
        os.path.dirname(config['model_save_path'])
    ]
    
    for path in required_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Path does not exist: {path}")

def train():
    # Configuration
    config = {
        'batch_size': 8,
        'epochs': 50,
        'learning_rate': 1e-4,
        'tile_size': 256,
        'train_image_dir': os.path.join('dataset', 'images_tiled'),
        'train_mask_dir': os.path.join('dataset', 'masks_tiled'),
        'val_split': 0.2,
        'model_save_path': os.path.join('model', 'unet_model.h5')
    }

    try:
        # Verify paths exist
        verify_paths(config)
        
        # Initialize data loader
        print("Loading dataset...")
        full_loader = SatelliteDataLoader(
            image_dir=config['train_image_dir'],
            mask_dir=config['train_mask_dir'],
            batch_size=config['batch_size'],
            tile_size=config['tile_size'],
            shuffle=True
        )

        # Split into train/validation
        print("Splitting dataset...")
        train_loader, val_loader = full_loader.split(val_ratio=config['val_split'])
        
        print(f"\nDataset Summary:")
        print(f"- Total samples: {len(full_loader.image_paths)}")
        print(f"- Training samples: {len(train_loader.image_paths)}")
        print(f"- Validation samples: {len(val_loader.image_paths)}")
        print(f"- Batch size: {config['batch_size']}")
        print(f"- Input shape: ({config['tile_size']}, {config['tile_size']}, 3)\n")

        # Initialize model
        print("Initializing U-Net model...")
        if os.path.exists(config['model_save_path']):
            print("Found existing model. Loading...")
            model = tf.keras.models.load_model(config['model_save_path'], custom_objects={'IoU': tf.keras.metrics.IoU})
        else:
            print("No existing model found. Initializing new model...")
            model = unet_model(input_size=(config['tile_size'], config['tile_size'], 3))
            model.compile(
                optimizer=Adam(config['learning_rate']),
                loss='binary_crossentropy',
                metrics=[
                'accuracy',
                tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1], name='iou')
                ]
            )


        model.summary()

        # Callbacks
        callbacks = [
            ModelCheckpoint(
                config['model_save_path'],
                monitor='val_iou',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]

        # Train model
        print("\nStarting training...")
        history = model.fit(
            train_loader,
            validation_data=val_loader,
            epochs=config['epochs'],
            callbacks=callbacks,
            verbose=1
        )

        # Save final model
        final_model_path = config['model_save_path'].replace('.h5', '_final.h5')
        model.save(final_model_path)
        print(f"\nSaved final model to: {final_model_path}")

        # Plot training history
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history['iou'], label='Train IoU')
        plt.plot(history.history['val_iou'], label='Validation IoU')
        plt.title('Training and Validation IoU')
        plt.xlabel('Epoch')
        plt.ylabel('IoU')
        plt.legend()

        plt.tight_layout()
        history_plot_path = os.path.join('model', 'training_history.png')
        plt.savefig(history_plot_path)
        print(f"Saved training history plot to: {history_plot_path}")
        plt.show()

    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("Please verify:")
        print(f"1. Dataset exists at {config['train_image_dir']} and {config['train_mask_dir']}")
        print(f"2. Images and masks are properly paired")
        print(f"3. File extensions are correct (.png, .jpg, .tif)")
        raise

if __name__ == '__main__':
    # Configure GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            print(e)

    train()