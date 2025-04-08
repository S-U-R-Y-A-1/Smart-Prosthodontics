from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def analyze_detections(model, image_path):
    # Load and preprocess image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image at {image_path}")
        return
    
    # Predict with two confidence thresholds
    results = model.predict(img, conf=0.3, save=False)
    
    if len(results[0].boxes) == 0:
        # Fallback with lower confidence
        results = model.predict(img, conf=0.2, save=False)
        if len(results[0].boxes) == 0:
            # Final fallback with augmentation
            results = model.predict(img, conf=0.15, augment=True, save=False)
    
    # Visualization
    if len(results[0].boxes) > 0:
        im_array = results[0].plot()
        title = f"Detected {len(results[0].boxes)} teeth"
    else:
        im_array = img
        title = "No detections after multiple attempts"
    
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()
    
    # Print detection summary
    if len(results[0].boxes) > 0:
        detections = Counter([model.names[int(box.cls)] for box in results[0].boxes])
        print("Detection Summary:")
        for cls, count in detections.items():
            print(f"- {cls}: {count}")
    else:
        print("No teeth detected even with low confidence threshold (0.15)")
        print("Possible solutions:")
        print("1. Check if the image type matches your training data")
        print("2. Verify your model was trained on similar panoramic X-rays")
        print("3. Consider adding more training examples")

# Load model
model = YOLO('D:/gen/runs/detect/train/weights/best.pt')

# Analyze specific image
analyze_detections(model, 'D:/gen/dataset/imgs/val/483.jpg')

