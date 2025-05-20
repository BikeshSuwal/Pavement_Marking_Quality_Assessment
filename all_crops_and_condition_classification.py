import os
import cv2
import numpy as np
import pandas as pd
import random
import psutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import concurrent.futures
from functools import partial
import gc

import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F

from segment_anything import sam_model_registry, SamPredictor

# === CONFIG ===
image_dir = Path("output")
label_dir = Path("runs/detect/predict/labels")
crop_dir = Path("crops")
mask_dir = Path("masks")
metadata_csv = Path("crops_metadata.csv")
checkpoint_path = "models/sam_vit_b_01ec64.pth"

# === PARALLELIZATION CONFIG ===
# Reduced worker count to prevent memory overload
MAX_WORKERS = 2

# === MEMORY MANAGEMENT CONFIG ===
# Process files in batches rather than all at once
BATCH_SIZE = 10  # Process this many files at a time
# Maximum number of images to keep in cache at once
MAX_CACHE_SIZE = 5

crop_dir.mkdir(parents=True, exist_ok=True)
mask_dir.mkdir(parents=True, exist_ok=True)

crop_size = 128
num_patches_per_image = 3
metadata = []

# === Device Configuration ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8  # Adjust based on your GPU memory

# === Memory monitoring utilities ===
def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_gpu_memory():
    """Get GPU memory usage if available"""
    if device.type == 'cuda':
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def print_memory_usage():
    """Print current memory usage"""
    ram = get_memory_usage()
    gpu = get_gpu_memory()
    print(f"RAM Usage: {ram:.2f} MB | GPU Memory: {gpu:.2f} MB")

# === Load Models with memory optimization ===
def load_sam_model():
    """Load SAM model with memory optimization"""
    # Clear CUDA cache first
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    print("Loading SAM model...")
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    print("SAM model loaded")
    return predictor

def load_classifier():
    """Load classifier model"""
    NUM_CLASSES = 3
    LABELS = {0: 'good', 1: 'damaged', 2: 'missing'}
    
    print("Loading classifier model...")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load("condition_classifier_resnet18.pth", map_location=device))
    model.to(device)
    model.eval()
    print("Classifier loaded")
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])
    
    return model, transform, LABELS

# LRU Cache for images instead of preloading all
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}
        self.order = []
    
    def get(self, key):
        if key in self.cache:
            # Move to end (most recently used)
            self.order.remove(key)
            self.order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if key in self.cache:
            # Update existing
            self.cache[key] = value
            self.order.remove(key)
            self.order.append(key)
        else:
            # Add new, check capacity
            if len(self.order) >= self.capacity:
                # Remove least recently used
                old_key = self.order.pop(0)
                del self.cache[old_key]
            self.cache[key] = value
            self.order.append(key)
    
    def clear(self):
        self.cache = {}
        self.order = []

def batch_classify_crops(model, transform, labels, image_paths):
    """Classify multiple crops in a batch for better GPU utilization"""
    if not image_paths:
        return []
        
    batch = torch.zeros((len(image_paths), 3, 128, 128), device=device)
    
    for i, path in enumerate(image_paths):
        image = Image.open(path).convert('RGB')
        batch[i] = transform(image)
    
    with torch.no_grad():
        outputs = model(batch)
        probs = F.softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, 1)
    
    results = []
    for i in range(len(image_paths)):
        results.append((labels[predicted[i].item()], conf[i].item()))
    
    return results

def calculate_coverage_ratio(mask):
    """Calculate coverage ratio from mask array directly"""
    if mask is None:
        return 0.0
    total_pixels = mask.size
    white_pixels = np.count_nonzero(mask > 127)
    return white_pixels / total_pixels

def yolo_to_pixel_coords(box, img_w, img_h):
    """Convert YOLO format to pixel coordinates"""
    xc, yc, bw, bh = box
    x_min = int((xc - bw / 2) * img_w)
    y_min = int((yc - bh / 2) * img_h)
    x_max = int((xc + bw / 2) * img_w)
    y_max = int((yc + bh / 2) * img_h)
    return [x_min, y_min, x_max, y_max]

def is_road_patch(patch):
    """Check if a patch is likely a road surface"""
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return hsv[..., 1].mean() < 50

def process_label_file(label_path, predictor, classifier_model, transform, labels, image_cache):
    """Process a single label file"""
    local_metadata = []
    image_name = label_path.stem + ".jpg"
    
    # Get image from cache or load it
    image = image_cache.get(image_name)
    if image is None:
        image_path = image_dir / image_name
        if not image_path.exists():
            print(f"❌ Missing image for {label_path.name}")
            return local_metadata
        image = cv2.imread(str(image_path))
        image_cache.put(image_name, image)
    
    h, w, _ = image.shape
    predictor.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    with open(label_path, "r") as f:
        detections = [line.strip().split() for line in f if len(line.strip().split()) in [5, 6]]

    if not detections:
        # Clear image from predictor to save memory
        predictor.reset_image()
        return local_metadata

    crops_to_classify = []
    crop_paths = []
    temp_data = []

    for i, det in enumerate(detections):
        if len(det) == 5:
            class_id, xc, yc, bw, bh = map(float, det)
            yolo_confidence = 0.9
        else:
            class_id, xc, yc, bw, bh, yolo_confidence = map(float, det)

        box = yolo_to_pixel_coords([xc, yc, bw, bh], w, h)
        input_box = np.array(box)

        # Get SAM mask
        masks, _, _ = predictor.predict(box=input_box, multimask_output=False)
        mask = masks[0].astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Ensure mask is properly sized to match the input image
        h, w = image.shape[:2]
        if mask_clean.shape[:2] != (h, w):
            mask_clean = cv2.resize(mask_clean, (w, h))

        # Save binary mask
        mask_name = f"{label_path.stem}_{i}.png"
        mask_path = mask_dir / mask_name
        cv2.imwrite(str(mask_path), mask_clean)

        # Get bounding rect of mask
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        x, y, box_w, box_h = cv2.boundingRect(contours[0])
        if box_w < 10 or box_h < 10:
            continue

        masked = cv2.bitwise_and(image, image, mask=mask_clean)
        crop = masked[y:y+box_h, x:x+box_w]
        crop_name = f"{label_path.stem}_{i}.png"
        crop_path = crop_dir / crop_name
        cv2.imwrite(str(crop_path), crop)

        # Save data for later batch processing
        crops_to_classify.append(crop_path)
        crop_paths.append(str(crop_path).replace("\\", "/"))
        coverage = calculate_coverage_ratio(mask_clean)
        
        temp_data.append({
            "original_image": image_name,
            "mask_file": mask_name,
            "crop_file": crop_paths[-1],
            "x": x, "y": y, "width": box_w, "height": box_h,
            "yolo_confidence": yolo_confidence,
            "coverage": coverage
        })
    
    # Clear image from predictor to save memory
    predictor.reset_image()
    
    # Batch classify all crops from this label file
    if crops_to_classify:
        # Process in smaller batches if needed
        batch_results = []
        for i in range(0, len(crops_to_classify), batch_size):
            batch = crops_to_classify[i:i+batch_size]
            batch_results.extend(batch_classify_crops(classifier_model, transform, labels, batch))
        
        # Add classification results to metadata
        for i, (predicted_label, classifier_confidence) in enumerate(batch_results):
            quality_score = {"good": 1.0, "damaged": 0.5, "missing": 0.0}[predicted_label]
            temp_data[i]["classifier_confidence"] = classifier_confidence
            temp_data[i]["predicted_condition"] = predicted_label
            temp_data[i]["quality_score"] = quality_score
            local_metadata.append(temp_data[i])
    
    return local_metadata

def process_unlabeled_image(img_path, classifier_model, transform, labels, image_cache):
    """Process an image without labels by adding random patches"""
    local_metadata = []
    
    # Get image from cache or load it
    img = image_cache.get(img_path.name)
    if img is None:
        img = cv2.imread(str(img_path))
        image_cache.put(img_path.name, img)
    
    h, w, _ = img.shape
    patches_added, attempts = 0, 0
    
    crop_paths = []
    temp_data = []
    
    while patches_added < num_patches_per_image and attempts < 30:
        x = random.randint(0, w - crop_size)
        y = random.randint(0, h - crop_size)
        patch = img[y:y+crop_size, x:x+crop_size]

        if is_road_patch(patch):
            crop_name = f"{img_path.stem}_missing_{patches_added}.png"
            crop_path = crop_dir / crop_name
            cv2.imwrite(str(crop_path), patch)
            
            crop_paths.append(crop_path)
            temp_data.append({
                "original_image": img_path.name,
                "mask_file": "NONE",
                "crop_file": str(crop_path).replace("\\", "/"),
                "x": x, "y": y, "width": crop_size, "height": crop_size,
                "yolo_confidence": 0.9,
                "coverage": 0.0  # No mask = 0 coverage
            })
            
            patches_added += 1
        attempts += 1
    
    # Batch classify crops
    if crop_paths:
        batch_results = batch_classify_crops(classifier_model, transform, labels, crop_paths)
        
        for i, (predicted_label, classifier_confidence) in enumerate(batch_results):
            quality_score = {"good": 1.0, "damaged": 0.5, "missing": 0.0}[predicted_label]
            temp_data[i]["classifier_confidence"] = classifier_confidence
            temp_data[i]["predicted_condition"] = predicted_label
            temp_data[i]["quality_score"] = quality_score
            local_metadata.append(temp_data[i])
    
    return local_metadata


def process_in_batches(files, process_func, predictor, classifier_model, transform, labels):
    """Process files in batches to manage memory usage"""
    all_results = []
    
    # Create LRU cache for images
    image_cache = LRUCache(MAX_CACHE_SIZE)
    
    for i in range(0, len(files), BATCH_SIZE):
        batch = files[i:i+BATCH_SIZE]
        print(f"Processing batch {i//BATCH_SIZE + 1}/{(len(files)-1)//BATCH_SIZE + 1} ({len(batch)} files)")
        
        batch_results = []
        
        # Process label files sequentially if they need the SAM predictor
        if process_func == process_label_file:
            # Process sequentially to avoid race conditions with predictor
            batch_results = []
            for file_path in tqdm(batch):
                result = process_func(
                    file_path,
                    predictor=predictor,
                    classifier_model=classifier_model,
                    transform=transform,
                    labels=labels,
                    image_cache=image_cache
                )
                batch_results.append(result)
        else:
            # For unlabeled images, we can still use parallel processing
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                func = partial(process_func,
                              classifier_model=classifier_model,
                              transform=transform,
                              labels=labels,
                              image_cache=image_cache)
                
                batch_results = list(tqdm(executor.map(func, batch), total=len(batch)))
        
        # Collect results
        for result in batch_results:
            all_results.extend(result)
        
        # Clear cache between batches to free memory
        image_cache.clear()
        
        # Force garbage collection
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
        # Print memory usage
        print_memory_usage()
    
    return all_results

def main():
    print("Starting road analysis with memory optimizations...")
    print_memory_usage()
    
    # Load models
    predictor = load_sam_model()
    classifier_model, transform, labels = load_classifier()
    
    print_memory_usage()
    
    all_metadata = []
    
    # Process all label files in batches
    label_files = sorted(label_dir.glob("*.txt"))
    print(f"Processing {len(label_files)} label files in batches...")
    
    if label_files:
        results = process_in_batches(
            label_files, 
            process_label_file,
            predictor,
            classifier_model, 
            transform, 
            labels
        )
        all_metadata.extend(results)
    
    # Get all images that have already been processed
    labelled_images = set([entry["original_image"] for entry in all_metadata])
    unlabeled_images = [img_path for img_path in sorted(image_dir.glob("*.jpg")) 
                      if img_path.name not in labelled_images]
    
    if unlabeled_images:
        print(f"Processing {len(unlabeled_images)} unlabeled images in batches...")
        
        # Process unlabeled images in batches
        results = process_in_batches(
            unlabeled_images, 
            process_unlabeled_image,
            predictor,
            classifier_model, 
            transform, 
            labels
        )
        all_metadata.extend(results)
    
    # Save metadata
    if all_metadata:
        df = pd.DataFrame(all_metadata)
        df.to_csv(metadata_csv, index=False)
        print(f"\n✅ Total crops: {len(df)}")
        print(f"🖼️  Saved to: {crop_dir}")
        print(f"📄 Metadata CSV: {metadata_csv}")
    else:
        print("⚠️ No crops generated.")

if __name__ == "__main__":
    main()