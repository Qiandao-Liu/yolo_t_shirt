"""
YOLOv8 training script - T-Shirt & Sleeve detection and tracking
Optimized for challenging cases where sleeves are inside the t-shirt
"""

import os
import yaml
import shutil
import random
from pathlib import Path
from ultralytics import YOLO
import torch
import argparse


class TShirtSleeveTrainer:
    """
    YOLO trainer for T-shirt and Sleeve detection
    Optimized for challenging cases with sleeves nested inside t-shirts
    """
    
    def __init__(self, data_dir, output_dir="./runs"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Class definitions
        self.classes = ['t-shirt', 'sleeve']
        
        # Check GPU
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            
    def prepare_dataset(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """
        Prepare the dataset and split train/val/test
        """
        print("\n" + "="*50)
        print("Preparing dataset...")
        print("="*50)
        
        # Check source data
        images_dir = self.data_dir
        labels_dir = self.data_dir / "labels"
        
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")
        if not labels_dir.exists():
            raise FileNotFoundError(f"Labels directory not found: {labels_dir}")
            
        # Collect all labeled images
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        all_images = []
        
        for img_file in images_dir.iterdir():
            if img_file.suffix.lower() in image_extensions:
                label_file = labels_dir / (img_file.stem + '.txt')
                if label_file.exists():
                    all_images.append(img_file.name)
                    
        print(f"Found {len(all_images)} labeled images")
        
        if len(all_images) < 10:
            raise ValueError("Too few labeled images; at least 10 are required for training")
            
        # Shuffle
        random.shuffle(all_images)
        
        # Split dataset
        n_train = int(len(all_images) * train_ratio)
        n_val = int(len(all_images) * val_ratio)
        
        train_images = all_images[:n_train]
        val_images = all_images[n_train:n_train + n_val]
        test_images = all_images[n_train + n_val:]
        
        print(f"Train: {len(train_images)} images")
        print(f"Validation: {len(val_images)} images")
        print(f"Test: {len(test_images)} images")
        
        # Create dataset directory structure
        dataset_dir = self.output_dir / "dataset"
        for split in ['train', 'val', 'test']:
            (dataset_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (dataset_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
            
        # Copy files
        def copy_files(image_list, split_name):
            for img_name in image_list:
                src_img = images_dir / img_name
                src_label = labels_dir / (Path(img_name).stem + '.txt')
                
                dst_img = dataset_dir / split_name / 'images' / img_name
                dst_label = dataset_dir / split_name / 'labels' / (Path(img_name).stem + '.txt')
                
                shutil.copy2(src_img, dst_img)
                shutil.copy2(src_label, dst_label)
                
        copy_files(train_images, 'train')
        copy_files(val_images, 'val')
        copy_files(test_images, 'test')
        
        # Create YOLO-format data.yaml
        data_yaml = {
            'path': str(dataset_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': {i: name for i, name in enumerate(self.classes)},
            'nc': len(self.classes)
        }
        
        yaml_path = dataset_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
            
        print(f"\nDataset config saved to: {yaml_path}")
        return yaml_path
        
    def create_training_config(self):
        """
        Create training config for challenging detection scenarios
        """
        # High-intensity training config for sleeves nested in t-shirts
        config = {
            # Model config
            'model': 'yolov8x.pt',  # Use the largest model for best performance
            
            # Training hyperparameters - high intensity
            'epochs': 300,          # Many epochs to fully learn
            'patience': 50,         # Early stopping patience
            'batch': -1,            # Auto-select best batch size
            'imgsz': 640,           # Image size
            
            # Optimizer settings
            'optimizer': 'AdamW',   # Use AdamW optimizer
            'lr0': 0.001,           # Initial learning rate
            'lrf': 0.01,            # Final LR factor
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 5,     # Warmup epochs
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            
            # Loss weights
            'box': 7.5,             # Box loss weight
            'cls': 0.5,             # Class loss weight
            'dfl': 1.5,             # DFL loss weight
            
            # Data augmentation - strong augmentation for generalization
            'hsv_h': 0.015,         # Hue shift
            'hsv_s': 0.7,           # Saturation shift
            'hsv_v': 0.4,           # Value shift
            'degrees': 15,          # Rotation
            'translate': 0.1,       # Translation
            'scale': 0.5,           # Scale
            'shear': 5,             # Shear
            'perspective': 0.0005,  # Perspective
            'flipud': 0.0,          # Vertical flip
            'fliplr': 0.5,          # Horizontal flip
            'mosaic': 1.0,          # Mosaic augmentation
            'mixup': 0.15,          # MixUp augmentation
            'copy_paste': 0.3,      # Copy-paste augmentation
            
            # Other settings
            'close_mosaic': 20,     # Disable mosaic for last N epochs
            'amp': True,            # Mixed precision training
            'fraction': 1.0,        # Use full dataset
            'profile': False,
            'overlap_mask': True,
            'mask_ratio': 4,
            'dropout': 0.1,         # Dropout regularization
            
            # Validation settings
            'val': True,
            'save': True,
            'save_period': 10,      # Save every 10 epochs
            'plots': True,          # Generate training plots
            'verbose': True,
        }
        
        return config
        
    def train(self, data_yaml_path, resume=False, pretrained_weights=None):
        """
        Run training
        """
        print("\n" + "="*50)
        print("Starting YOLOv8 training")
        print("="*50)
        
        config = self.create_training_config()
        
        # Select model
        if pretrained_weights and os.path.exists(pretrained_weights):
            print(f"Loading pretrained weights: {pretrained_weights}")
            model = YOLO(pretrained_weights)
        elif resume:
            # Find latest checkpoint
            last_weight = self.output_dir / "train" / "weights" / "last.pt"
            if last_weight.exists():
                print(f"Resuming training: {last_weight}")
                model = YOLO(str(last_weight))
            else:
                print("No checkpoint found, training from scratch")
                model = YOLO(config['model'])
        else:
            print(f"Using pretrained model: {config['model']}")
            model = YOLO(config['model'])
            
        # Start training
        results = model.train(
            data=str(data_yaml_path),
            epochs=config['epochs'],
            patience=config['patience'],
            batch=config['batch'],
            imgsz=config['imgsz'],
            
            # Optimizer
            optimizer=config['optimizer'],
            lr0=config['lr0'],
            lrf=config['lrf'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay'],
            warmup_epochs=config['warmup_epochs'],
            warmup_momentum=config['warmup_momentum'],
            warmup_bias_lr=config['warmup_bias_lr'],
            
            # Loss weights
            box=config['box'],
            cls=config['cls'],
            dfl=config['dfl'],
            
            # Data augmentation
            hsv_h=config['hsv_h'],
            hsv_s=config['hsv_s'],
            hsv_v=config['hsv_v'],
            degrees=config['degrees'],
            translate=config['translate'],
            scale=config['scale'],
            shear=config['shear'],
            perspective=config['perspective'],
            flipud=config['flipud'],
            fliplr=config['fliplr'],
            mosaic=config['mosaic'],
            mixup=config['mixup'],
            copy_paste=config['copy_paste'],
            
            # Other
            close_mosaic=config['close_mosaic'],
            amp=config['amp'],
            dropout=config['dropout'],
            
            # Output settings
            project=str(self.output_dir),
            name='train',
            exist_ok=True,
            save=config['save'],
            save_period=config['save_period'],
            plots=config['plots'],
            verbose=config['verbose'],
            device=self.device,
        )
        
        print("\nTraining complete!")
        print(f"Best model saved at: {self.output_dir}/train/weights/best.pt")
        
        return results
        
    def validate(self, weights_path=None):
        """
        Validate model performance
        """
        print("\n" + "="*50)
        print("Validating model")
        print("="*50)
        
        if weights_path is None:
            weights_path = self.output_dir / "train" / "weights" / "best.pt"
            
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
            
        model = YOLO(str(weights_path))
        
        # Evaluate on validation set
        results = model.val(
            project=str(self.output_dir),
            name='validate',
            exist_ok=True,
            verbose=True,
        )
        
        return results
        
    def export(self, weights_path=None, formats=['onnx', 'torchscript']):
        """
        Export model to multiple formats
        """
        print("\n" + "="*50)
        print("Exporting model")
        print("="*50)
        
        if weights_path is None:
            weights_path = self.output_dir / "train" / "weights" / "best.pt"
            
        model = YOLO(str(weights_path))
        
        exported_paths = []
        for fmt in formats:
            print(f"\nExporting to {fmt} format...")
            path = model.export(format=fmt, device=self.device)
            exported_paths.append(path)
            print(f"Exported: {path}")
            
        return exported_paths


class TShirtSleeveTracker:
    """
    Real-time tracker for T-shirt and Sleeve
    """
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.classes = ['t-shirt', 'sleeve']
        
    def track_video(self, source, output_path=None, show=True, conf=0.25, iou=0.45):
        """
        Track on a video file or camera
        
        Args:
            source: Video file path or camera ID (0 for default camera)
            output_path: Output video path
            show: Whether to show live view
            conf: Confidence threshold
            iou: IoU threshold
        """
        print("\n" + "="*50)
        print("Starting tracking")
        print("="*50)
        
        # Track with BoT-SORT tracker
        results = self.model.track(
            source=source,
            persist=True,           # Persist track IDs
            tracker="botsort.yaml", # Use BoT-SORT tracker
            conf=conf,
            iou=iou,
            show=show,
            save=output_path is not None,
            project=os.path.dirname(output_path) if output_path else "./runs/track",
            name=os.path.basename(output_path).split('.')[0] if output_path else "track",
            stream=True,            # Stream processing
            verbose=True,
        )
        
        # Process results
        for frame_id, result in enumerate(results):
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confs = result.boxes.conf.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int) if result.boxes.id is not None else None
                
                print(f"\nFrame {frame_id}:")
                for i in range(len(boxes)):
                    class_name = self.classes[classes[i]]
                    conf = confs[i]
                    track_id = track_ids[i] if track_ids is not None else "N/A"
                    print(f"  - {class_name} (ID: {track_id}, confidence: {conf:.2f})")
                    
    def track_realsense(self, output_path=None, duration=None, conf=0.25, iou=0.45):
        """
        Real-time tracking with a RealSense camera
        
        Args:
            output_path: Output video path
            duration: Tracking duration in seconds, None for unlimited
            conf: Confidence threshold
            iou: IoU threshold
        """
        import pyrealsense2 as rs
        import cv2
        import time
        
        print("\n" + "="*50)
        print("RealSense real-time tracking")
        print("="*50)
        
        # Configure RealSense
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        
        pipeline.start(config)
        
        # Video writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30.0, (1280, 720))
            
        start_time = time.time()
        frame_count = 0
        
        try:
            while True:
                # Check time limit
                if duration and (time.time() - start_time) > duration:
                    break
                    
                # Get frame
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    continue
                    
                frame = np.asanyarray(color_frame.get_data())
                frame_count += 1
                
                # Track
                results = self.model.track(
                    source=frame,
                    persist=True,
                    tracker="botsort.yaml",
                    conf=conf,
                    iou=iou,
                    verbose=False,
                )
                
                # Draw results
                annotated_frame = results[0].plot()
                
                # Add FPS overlay
                fps = frame_count / (time.time() - start_time)
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Display
                cv2.imshow("T-Shirt & Sleeve Tracking", annotated_frame)
                
                # Save
                if out:
                    out.write(annotated_frame)
                    
                # Press q to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            pipeline.stop()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
        print(f"\nTracking complete! Processed {frame_count} frames")


def main():
    parser = argparse.ArgumentParser(description='T-Shirt & Sleeve YOLO training and tracking')
    parser.add_argument('--mode', type=str, required=True, 
                        choices=['prepare', 'train', 'validate', 'export', 'track'],
                        help='Run mode')
    parser.add_argument('--data', type=str, default='./captured_images',
                        help='Data directory path')
    parser.add_argument('--output', type=str, default='./runs',
                        help='Output directory path')
    parser.add_argument('--weights', type=str, default=None,
                        help='Pretrained weights path')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training')
    parser.add_argument('--source', type=str, default='0',
                        help='Tracking source (file path or camera ID)')
    parser.add_argument('--track-output', type=str, default=None,
                        help='Tracking output path')
    
    args = parser.parse_args()
    
    if args.mode in ['prepare', 'train', 'validate', 'export']:
        trainer = TShirtSleeveTrainer(args.data, args.output)
        
        if args.mode == 'prepare':
            trainer.prepare_dataset()
            
        elif args.mode == 'train':
            # Prepare the dataset first
            data_yaml = trainer.prepare_dataset()
            # Then train
            trainer.train(data_yaml, resume=args.resume, pretrained_weights=args.weights)
            
        elif args.mode == 'validate':
            trainer.validate(args.weights)
            
        elif args.mode == 'export':
            trainer.export(args.weights)
            
    elif args.mode == 'track':
        if args.weights is None:
            args.weights = os.path.join(args.output, 'train', 'weights', 'best.pt')
            
        tracker = TShirtSleeveTracker(args.weights)
        
        # Determine video file vs camera
        try:
            source = int(args.source)  # Camera ID
        except ValueError:
            source = args.source  # Video file path
            
        tracker.track_video(source, args.track_output)


if __name__ == "__main__":
    main()
    
    
# ================== Usage Examples ==================
"""
# 1. Prepare the dataset
python 3_yolo_trainer.py --mode prepare --data ./captured_images

# 2. Train the model
python 3_yolo_trainer.py --mode train --data ./captured_images --output ./runs

# 3. Resume training from a checkpoint
python 3_yolo_trainer.py --mode train --data ./captured_images --resume

# 4. Validate the model
python 3_yolo_trainer.py --mode validate --weights ./runs/train/weights/best.pt

# 5. Export the model
python 3_yolo_trainer.py --mode export --weights ./runs/train/weights/best.pt

# 6. Track with a camera
python 3_yolo_trainer.py --mode track --weights ./runs/train/weights/best.pt --source 0

# 7. Track a video file
python 3_yolo_trainer.py --mode track --weights ./runs/train/weights/best.pt --source video.mp4 --track-output result.mp4
"""
