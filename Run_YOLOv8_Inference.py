from ultralytics import YOLO
import os
from pathlib import Path

# Load your trained model
model = YOLO('best.pt')  # path to your trained weights

# Folder with test images
image_folder = 'output'

# Run inference on all images, saving images and labels
results = model.predict(source=image_folder, save=True, save_txt=True, save_conf=True, conf=0.25)

# Folder where YOLO saves label files by default (check your ultralytics version)
# Usually, it saves in runs/detect/predict/labels
results_folder = results[0].save_dir  # get save directory from results object
label_folder = os.path.join(results_folder, 'labels')

# Ensure label folder exists
os.makedirs(label_folder, exist_ok=True)

# Get list of all image filenames in test folder
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# For each image, create an empty label file if none exists (means no detections)
for img_name in image_files:
    label_name = Path(img_name).stem + '.txt'
    label_path = os.path.join(label_folder, label_name)
    if not os.path.exists(label_path):
        with open(label_path, 'w') as f:
            pass  # create empty label file
