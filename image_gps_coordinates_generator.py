import pandas as pd
from pathlib import Path

# Load metadata
metadata = pd.read_csv("image_metadata.csv")

# Extract image ID from file_path column
metadata["image_id"] = metadata["file_path"].apply(
    lambda x: Path(x).stem  # e.g., 'mapillary_images/437859.jpg' → '437859'
)

# Keep only necessary columns
metadata = metadata[["image_id", "latitude", "longitude"]]

# Load crop label data
crops_df = pd.read_csv("crops_metadata.csv")

# Extract image_id from crop_file (e.g., 'datasets/.../123456789_0.png' → '123456789')
crops_df["image_id"] = crops_df["crop_file"].apply(lambda x: Path(x).stem.split("_")[0])

# Merge GPS data using image_id
merged = crops_df.merge(metadata, on="image_id", how="left")

# Output: file with GPS coordinates per crop
merged[["crop_file", "latitude", "longitude"]].to_csv(
    "datasets/split_for_labeling/train/image_gps_coordinates.csv", index=False
)

print("✅ image_gps_coordinates.csv successfully created and matched with GPS data.")
