from ultralytics import YOLO
from multiprocessing import freeze_support

def main():
    # 1. Verify environment
    print("Starting dental teeth detection training...")
    
    # 2. Load model (consider using larger model like 'yolov8m.pt' if better accuracy is needed)
    model = YOLO('yolov8m.pt')  # or 'yolov8s.pt', 'yolov8m.pt' for better accuracy
    
    # 3. Train with optimized settings for dental detection
    try:
        results = model.train(
            data='data.yaml',
            epochs=100,            # Increased epochs for better convergence
            imgsz=640,            # Higher resolution for small teeth
            batch=8,              # Adjusted batch size (reduce if OOM errors occur)
            device='cpu',         # Change to '0' if GPU available
            workers=2,            # Can increase if not on CPU
            single_cls=False,     # Keep False for multi-class (32 teeth)
            verbose=True,
            
            # Additional recommended parameters:
            lr0=0.01,            # Initial learning rate
            lrf=0.01,            # Final learning rate
            momentum=0.937,       # SGD momentum
            weight_decay=0.0005,  # Optimizer weight decay
            dropout=0.1,          # Regularization (if using larger model)
            label_smoothing=0.1,  # Helps with class imbalance
            
            # Augmentation settings important for dental images:
            hsv_h=0.015,         # Hue augmentation
            hsv_s=0.7,           # Saturation augmentation
            hsv_v=0.4,           # Value augmentation
            degrees=10,           # Rotation augmentation
            translate=0.1,        # Translation augmentation
            scale=0.5,           # Scale augmentation
            fliplr=0.5,          # Horizontal flip probability
            flipud=0.0,          # Vertical flip probability (often 0 for dental)
            
            # Tooth detection specific:
            overlap_mask=True,    # Important if teeth overlap
            mask_ratio=4,         # For segmentation if needed
            copy_paste=0.0        # Can be useful for data augmentation
        )
        
        # Save and export the model
        model.save('last_dental_teeth.pt')
        model.export(format='onnx')  # Optional: export to ONNX
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
    finally:
        print("Training completed. Results saved.")

if __name__ == '__main__':
    freeze_support()
    main()