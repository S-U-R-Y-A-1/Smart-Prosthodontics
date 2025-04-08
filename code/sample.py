import cv2
import numpy as np

# Load your image - replace 'your_image.jpg' with your image path
image_path = 'D:/gen/dataset/imgs/val/483.jpg'  # CHANGE THIS
img = cv2.imread(image_path)
if img is None:
    print("Error: Image not found! Please check the path.")
    exit()

# Get image dimensions
img_height, img_width = img.shape[:2]

# Define colors for visualization (BGR format)
COLORS = {
    1: (0, 0, 255),     # Red
    2: (0, 255, 0),     # Green
    5: (255, 0, 0),     # Blue
    10: (0, 255, 255),  # Yellow
    12: (255, 0, 255),  # Magenta
    14: (255, 255, 0),  # Cyan
    15: (0, 165, 255),  # Orange
    20: (128, 0, 128),  # Purple
    21: (0, 128, 128),  # Teal
    22: (128, 128, 0),  # Olive
    23: (192, 192, 192), # Silver
    24: (128, 0, 0),    # Maroon
    25: (0, 128, 0),    # Dark Green
    26: (0, 0, 128),    # Navy
    28: (128, 128, 128) # Gray
}

# Your label data
labels = [
    (12, 0.804263, 0.650391, 0.391475, 0.699219),
    (14, 0.833905, 0.662109, 0.332190, 0.675781),
    (10, 0.778050, 0.650391, 0.443900, 0.699219),
    (20, 0.786379, 0.775391, 0.427242, 0.449219),
    (21, 0.769476, 0.774902, 0.461049, 0.450195),
    (22, 0.756002, 0.781250, 0.487996, 0.437500),
    (23, 0.743753, 0.780273, 0.512494, 0.439453),
    (24, 0.731994, 0.786133, 0.536012, 0.427734),
    (26, 0.697452, 0.776855, 0.605096, 0.446289),
    (15, 0.858403, 0.647949, 0.283195, 0.704102),
    (25, 0.716316, 0.774902, 0.567369, 0.450195),
    (1, 0.607300, 0.645020, 0.785399, 0.709961),
    (2, 0.628123, 0.647949, 0.743753, 0.704102),
    (5, 0.695002, 0.631836, 0.609995, 0.736328),
    (28, 0.658501, 0.763184, 0.682999, 0.473633)
]
# Function to convert normalized coordinates to pixel values
def denormalize(x_center, y_center, width, height, img_width, img_height):
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    x1 = int(x_center - width/2)
    y1 = int(y_center - height/2)
    x2 = int(x_center + width/2)
    y2 = int(y_center + height/2)
    return x1, y1, x2, y2

# Draw each bounding box
for label in labels:
    class_id, x_center, y_center, width, height = label
    color = COLORS.get(class_id, (0, 0, 0))  # Default to black if class not defined
    
    # Convert to pixel coordinates
    x1, y1, x2, y2 = denormalize(x_center, y_center, width, height, img_width, img_height)
    
    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # Put class label and coordinates
    label_text = f"Class {class_id} | X:{x_center:.2f}, Y:{y_center:.2f}"
    cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

# Show the result
cv2.imshow('Image with Bounding Boxes', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the result
output_path = 'annotated_image.jpg'
cv2.imwrite(output_path, img)
print(f"Annotated image saved as {output_path}")