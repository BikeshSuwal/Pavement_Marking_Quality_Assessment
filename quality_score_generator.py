import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# === Load metadata ===
csv_path = "crops_metadata.csv"
df = pd.read_csv(csv_path)

# === Check required columns ===
required_columns = ["coverage", "yolo_confidence", "classifier_confidence", "predicted_condition"]
missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# === Normalise coverage and confidences ===
scaler = MinMaxScaler()
df[["coverage_norm", "yolo_conf_norm", "clf_conf_norm"]] = scaler.fit_transform(
    df[["coverage", "yolo_confidence", "classifier_confidence"]]
)

# === Condition weight mapping ===
condition_weights = {
    "good": 1.0,
    "damaged": 0.5,
    "missing": 0.0
}
df["condition_weight"] = df["predicted_condition"].map(condition_weights).fillna(0)

# === Compute quality score ===
df["quality_score"] = (
    0.4 * df["coverage_norm"] +
    0.3 * df["yolo_conf_norm"] +
    0.3 * df["clf_conf_norm"] * df["condition_weight"]
)

# === Save updated CSV ===
df.to_csv(csv_path, index=False)
print(f"✅ Updated CSV with quality_score saved to: {csv_path}")
