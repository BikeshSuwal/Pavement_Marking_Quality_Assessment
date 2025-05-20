import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import shutil
from tqdm import tqdm

# ==== CONFIG ====
input_dir = Path("mapillary_images")
output_dir = Path("outputs")
output_dir.mkdir(parents=True, exist_ok=True)

road_like_classes = [2, 3, 4, 6, 7, 8, 9]  # person, car, motorcycle, bus, train, truck, traffic light
threshold = 0.05  # Minimum % of image containing road-like pixels
max_input_size = 768  # Resize larger images to prevent GPU overload

# ==== DEVICE ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== MODEL ====
model = deeplabv3_resnet101(pretrained=True).to(device)
model.eval()

# ==== TRANSFORM ====
transform = T.Compose([
    T.Resize(max_input_size),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# ==== ROAD FILTER FUNCTION ====
def is_road_image_deeplab(image_bgr, threshold=threshold):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    predicted_classes = output.argmax(0).cpu().numpy()

    mask = np.isin(predicted_classes, road_like_classes)
    road_ratio = np.mean(mask)

    return road_ratio > threshold

# ==== PROCESS IMAGES ====
image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))

accepted, rejected = 0, 0
for img_path in tqdm(image_files, desc="Filtering road-like images"):
    img = cv2.imread(str(img_path))
    if img is None:
        rejected += 1
        continue

    try:
        if is_road_image_deeplab(img):
            shutil.copy(img_path, output_dir / img_path.name)
            accepted += 1
        else:
            rejected += 1
    except Exception as e:
        print(f"⚠️ Error processing {img_path.name}: {e}")
        rejected += 1

print(f"\n✅ Done! {accepted} images copied to {output_dir}.")
print(f"🚫 {rejected} images rejected or failed.")
