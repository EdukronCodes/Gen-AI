import os
import yaml
from ultralytics import YOLO
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_model(config_path, model_size='n'):
    """
    Train YOLOv8 model for retail shelf monitoring.
    
    Args:
        config_path (str): Path to YAML configuration file
        model_size (str): Model size ('n', 's', 'm', 'l', 'x')
    """
    try:
        # Load configuration
        config = load_config(config_path)
        logger.info("Configuration loaded successfully")

        # Initialize model
        model = YOLO(f'yolov8{model_size}.pt')
        logger.info(f"Initialized YOLOv8-{model_size} model")

        # Create results directory
        results_dir = Path('runs/train')
        results_dir.mkdir(parents=True, exist_ok=True)

        # Train model
        logger.info("Starting model training...")
        results = model.train(
            data=config_path,
            epochs=config['epochs'],
            batch=config['batch_size'],
            imgsz=config['imgsz'],
            device=config['device'],
            workers=config['workers'],
            patience=config['patience'],
            save=config['save'],
            save_period=config['save_period'],
            cache=config['cache'],
            exist_ok=config['exist_ok'],
            pretrained=config['pretrained'],
            optimizer=config['optimizer'],
            verbose=config['verbose'],
            seed=config['seed'],
            deterministic=config['deterministic'],
            single_cls=config['single_cls'],
            image_weights=config['image_weights'],
            rect=config['rect'],
            cos_lr=config['cos_lr'],
            close_mosaic=config['close_mosaic'],
            resume=config['resume'],
            amp=config['amp'],
            lr0=config['lr0'],
            lrf=config['lrf'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay'],
            warmup_epochs=config['warmup_epochs'],
            warmup_momentum=config['warmup_momentum'],
            warmup_bias_lr=config['warmup_bias_lr'],
            box=config['box'],
            cls=config['cls'],
            dfl=config['dfl'],
            fl_gamma=config['fl_gamma'],
            label_smoothing=config['label_smoothing'],
            nbs=config['nbs'],
            overlap_mask=config['overlap_mask'],
            mask_ratio=config['mask_ratio'],
            dropout=config['dropout'],
            val=config['val']
        )

        # Save model
        model_path = results_dir / 'best.pt'
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")

        # Validate model
        metrics = model.val()
        logger.info(f"Validation metrics: {metrics}")

        return model_path, metrics

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    config_path = "yolo_config.yaml"
    model_size = 'n'  # Change to 's', 'm', 'l', or 'x' for different model sizes
    
    try:
        model_path, metrics = train_model(config_path, model_size)
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {str(e)}") 