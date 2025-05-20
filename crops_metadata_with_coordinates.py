import pandas as pd
import geojson

# Load CSVs
crops_df = pd.read_csv("crops_metadata.csv")
image_df = pd.read_csv("image_metadata.csv")

# Clean 'original_image' by removing file extension
crops_df['original_image_clean'] = crops_df['original_image'].astype(str).str.replace(r'\.jpg$', '', regex=True)

# Ensure 'image_id' is string
image_df['image_id'] = image_df['image_id'].astype(str)

# Merge on cleaned fields
merged_df = crops_df.merge(
    image_df[['image_id', 'latitude', 'longitude']],
    left_on='original_image_clean',
    right_on='image_id',
    how='left'
)

# Drop helpers
merged_df = merged_df.drop(columns=['original_image_clean', 'image_id'])

# Save merged CSV
merged_df.to_csv("crops_metadata_with_coords.csv", index=False)
print("✅ CSV with coordinates saved: crops_metadata_with_coords.csv")

# --- Create GeoJSON ---
features = []
for _, row in merged_df.dropna(subset=['latitude', 'longitude']).iterrows():
    properties = row.to_dict()  # Keep lat/lon in properties too

    point = geojson.Point((row['longitude'], row['latitude']))
    feature = geojson.Feature(geometry=point, properties=properties)
    features.append(feature)

geojson_output = geojson.FeatureCollection(features)

# Save GeoJSON
with open("crops_metadata.geojson", "w") as f:
    geojson.dump(geojson_output, f)

print("✅ GeoJSON file saved: crops_metadata.geojson")
