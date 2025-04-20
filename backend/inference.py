import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt
import io
import os
from dotenv import load_dotenv

from PIL import ImageDraw, ImageFont



# Load environment variables
load_dotenv()


# === Load YOLOv11 model from Ultralytics ===
model = YOLO(os.getenv('MODEL_PATH', 'util/yolo11n.pt'))

# === Config ===
PATCH_SIZE = int(os.getenv('PATCH_SIZE', 640))
STRIDE = int(os.getenv('STRIDE', 80))
CONF_THRESHOLD = float(os.getenv('CONF_THRESHOLD', 0.40))
IOU_THRESHOLD = float(os.getenv('IOU_THRESHOLD', 0.45))

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






def generate_patch_grid(patch_images, max_cols=5, padding=10):
    """Create a single image showing all patches in a labeled grid."""
    font = ImageFont.load_default()
    cols = min(max_cols, len(patch_images))
    rows = (len(patch_images) + cols - 1) // cols

    patch_w, patch_h = patch_images[0].size
    grid_w = cols * patch_w + (cols - 1) * padding
    grid_h = rows * patch_h + (rows - 1) * padding + 20 * rows

    grid_image = Image.new("RGB", (grid_w, grid_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(grid_image)

    for idx, patch in enumerate(patch_images):
        row = idx // cols
        col = idx % cols
        x = col * (patch_w + padding)
        y = row * (patch_h + padding + 20)

        grid_image.paste(patch, (x, y + 20))
        draw.text((x + 5, y), f"Patch {idx}", fill=(0, 0, 0), font=font)

    return grid_image

def detect_people(image_input):
    """
    Returns:
        - patch_images: List of patch images with bounding boxes (PIL)
        - final_result_image: Merged detection result image (PIL)
        - patch_grid_image: Grid of all patches with boxes and titles (PIL)
    """
    image = np.array(image_input.convert("RGB"))
    patches, coords, padded_image, was_padded, original_size = create_patches(image, PATCH_SIZE, STRIDE)

    all_boxes = []
    all_scores = []
    patch_images = []

    for patch, (x_off, y_off) in zip(patches, coords):
        results = model.predict(patch)

        # Draw boxes on patch
        patch_boxes = []
        patch_scores = []

        for det in results[0].boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = det
            if conf < CONF_THRESHOLD:
                continue

            # Store for full-image merge
            x1_adj = x1 + x_off
            y1_adj = y1 + y_off
            x2_adj = x2 + x_off
            y2_adj = y2 + y_off
            all_boxes.append([x1_adj, y1_adj, x2_adj, y2_adj])
            all_scores.append(conf)

            # Store for patch drawing
            patch_boxes.append([x1, y1, x2, y2])
            patch_scores.append(conf)

        # Draw patch-level boxes
        patch_with_boxes = draw_boxes(patch.copy(), patch_boxes, patch_scores)
        patch_pil = Image.fromarray(patch_with_boxes)
        patch_images.append(patch_pil)

    # Final full image with all detections
    result_image = padded_image.copy()
    if len(all_boxes) > 0:
        result_image = draw_boxes(result_image, all_boxes, all_scores)
    if was_padded:
        h, w = original_size
        result_image = result_image[:h, :w]

    # Generate grid view of patches with boxes
    patch_grid_image = generate_patch_grid(patch_images)

    return patch_grid_image, Image.fromarray(result_image)
