import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import io
import os
from dotenv import load_dotenv
import torch
from ultralytics.nn.tasks import DetectionModel
from torch.serialization import add_safe_globals

# Add DetectionModel to safe globals
add_safe_globals([DetectionModel])

# Load environment variables
load_dotenv()

# === Load YOLOv11 model from Ultralytics ===
model_path = os.getenv('MODEL_PATH', 'yolo11n.pt')
model = YOLO(model_path)

# === Config ===
PATCH_SIZE = 640
STRIDE = 80
CONF_THRESHOLD = 0.40
IOU_THRESHOLD = 0.45

def create_patches(image, patch_size, stride):
    h, w, _ = image.shape
    padded = False
    original_size = (h, w)

    if h < patch_size or w < patch_size:
        pad_h = max(0, patch_size - h)
        pad_w = max(0, patch_size - w)
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=(0,0,0))
        h, w = image.shape[:2]
        padded = True

    patches, coords = [], []
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = image[y:y + patch_size, x:x + patch_size]
            patches.append(patch)
            coords.append((x, y))

    return patches, coords, image, padded, original_size

def draw_boxes(image, boxes, scores=None):
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        color = (0, 255, 0)  # Default green color
        thickness = 2
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        if scores is not None:
            conf = scores[i]
            cv2.putText(image, f'{conf:.2f}', (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
    return image

def detect_people(image_input):
    """
    Main detection function that processes an input image and returns both
    the patch visualization and the final detection result.
    
    Args:
        image_input (PIL.Image): Input image to process
        
    Returns:
        tuple: (patches_visualization, final_result_image)
    """
    image = np.array(image_input.convert("RGB"))
    patches, coords, padded_image, was_padded, original_size = create_patches(image, PATCH_SIZE, STRIDE)

    all_boxes = []
    all_scores = []
    print(f"Total patches: {len(patches)}")
    
    # Create a figure for displaying patches
    n_patches = len(patches)
    n_cols = min(5, n_patches)
    n_rows = (n_patches + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (patch, (x_off, y_off)) in enumerate(zip(patches, coords)):
        results = model.predict(patch)
        print(f"Processing patch {i}")

        # Create a copy of the patch for visualization
        patch_vis = patch.copy()
        
        for det in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = det

            if conf < CONF_THRESHOLD:
                continue

            # Draw box on patch visualization
            cv2.rectangle(patch_vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Adjust coordinates for full image
            x1_adj = x1 + x_off
            y1_adj = y1 + y_off
            x2_adj = x2 + x_off
            y2_adj = y2 + y_off
            
            all_boxes.append([x1_adj, y1_adj, x2_adj, y2_adj])
            all_scores.append(conf)
            
            print(f"Patch {i} detection: x1={x1:.2f}, y1={y1:.2f}, x2={x2:.2f}, y2={y2:.2f}, conf={conf:.2f}")
        
        # Display patch in matplotlib subplot
        row = i // n_cols
        col = i % n_cols
        axes[row, col].imshow(cv2.cvtColor(patch_vis, cv2.COLOR_BGR2RGB))
        axes[row, col].set_title(f'Patch {i}')
        axes[row, col].axis('off')

    # Hide empty subplots
    for i in range(len(patches), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
        
    plt.tight_layout()

       # Save figure to a temporary buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    plt.close()
    buf.seek(0)
    patches_image = Image.open(buf)

    # Create merged image with all detections
    result_image = padded_image.copy()
    if len(all_boxes) > 0:
        result_image = draw_boxes(result_image, all_boxes, all_scores)
    
    if was_padded:
        h, w = original_size
        result_image = result_image[:h, :w]

    return patches_image, Image.fromarray(result_image)