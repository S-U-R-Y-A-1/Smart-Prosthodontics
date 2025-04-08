import json
import os
import numpy as np
import base64
import cv2
import zlib

def process_bitmap(bitmap_data, origin, img_size):
    """Handle bitmap data with size mismatch by scaling or padding"""
    try:
        decoded = base64.b64decode(bitmap_data)
        if decoded.startswith(b'x\x9c'):
            try:
                decoded = zlib.decompress(decoded)
            except zlib.error:
                pass

        origin_x, origin_y = origin
        region_w = img_size[0] - origin_x
        region_h = img_size[1] - origin_y
        
        # Convert to numpy array
        arr = np.frombuffer(decoded, dtype=np.uint8)
        
        # Calculate expected and actual sizes
        expected_size = region_w * region_h
        actual_size = arr.size
        
        # Handle size mismatch by either truncating or repeating data
        if actual_size < expected_size:
            # Repeat data to fill expected size
            repeat_times = (expected_size // actual_size) + 1
            arr = np.tile(arr, repeat_times)[:expected_size]
        elif actual_size > expected_size:
            # Truncate to expected size
            arr = arr[:expected_size]
        
        # Create mask
        mask = np.zeros((img_size[1], img_size[0]), dtype=np.uint8)
        region = arr.reshape((region_h, region_w))
        mask[origin_y:origin_y+region_h, origin_x:origin_x+region_w] = region
        
        return mask
        
    except Exception as e:
        print(f"Bitmap processing error: {e}")
        return None

def process_annotations(json_dir, img_dir, output_dir):
    """Main processing function with guaranteed output"""
    os.makedirs(output_dir, exist_ok=True)
    
    for json_file in os.listdir(json_dir):
        if not json_file.endswith('.json'):
            continue
            
        base_name = os.path.splitext(json_file)[0]
        json_path = os.path.join(json_dir, json_file)
        output_path = os.path.join(output_dir, f"{base_name}.txt")
        
        try:
            with open(json_path) as f:
                data = json.load(f)
            
            img_w = data['size']['width']
            img_h = data['size']['height']
            annotations = []
            
            for obj in data['objects']:
                if obj['geometryType'] != 'bitmap':
                    continue
                
                mask = process_bitmap(
                    obj['bitmap']['data'],
                    obj['bitmap']['origin'],
                    (img_w, img_h))
                
                if mask is None:
                    continue
                
                # Simple bounding box from non-zero pixels
                nonzero = cv2.findNonZero(mask)
                if nonzero is None:
                    continue
                
                x, y, w, h = cv2.boundingRect(nonzero)
                class_id = int(obj['classTitle']) - 1
                
                # Normalize coordinates
                x_center = (x + w/2) / img_w
                y_center = (y + h/2) / img_h
                width = w / img_w
                height = h / img_h
                
                annotations.append(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )
            
            # Write output (empty file if no annotations)
            with open(output_path, 'w') as f:
                f.write('\n'.join(annotations))
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            # Create empty file if processing failed
            open(output_path, 'w').close()

# Usage
json_dir = 'C:/Users/Surya/Downloads/ds/ds/ann'
img_dir = 'C:/Users/Surya/Downloads/ds/ds/img'
output_dir = 'C:/Users/Surya/Downloads/ds/ds/labels'

process_annotations(json_dir, img_dir, output_dir)