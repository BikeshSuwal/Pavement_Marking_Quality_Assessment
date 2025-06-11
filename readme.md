# 🛣️ Automated Pavement Marking Quality Assessment Using Street-Level Images

This project provides a complete computer vision pipeline to **automatically assess the quality of dashed lane markings** using street-level imagery. The system combines **object detection (YOLOv8)**, **segmentation (SAM)**, and **classification (ResNet18)** to identify and evaluate road markings, and visualise their condition on an interactive map.

---
[![Watch Demo Video](https://github.com/user-attachments/assets/cd42bba7-6975-4e9d-9675-f449951e3a26)](https://github.com/user-attachments/assets/cd42bba7-6975-4e9d-9675-f449951e3a26)

---

## 📌 Objective

To detect, classify, and evaluate dashed road markings based on their **condition (Good / Damaged / Missing)**, and to compute a **quality score** for infrastructure maintenance and monitoring.

---

## ✅ Features

* 📸 Street-level image acquisition via **Mapillary API**
* ⚙️ Manual enhancement with **perspective correction**
* 🔍 Detection of markings using **YOLOv8**
* 🧹 Mask refinement using **SAM (Segment Anything Model)**
* ✂️ Marking and patch cropping for condition classification
* 🧠 Marking classification via **ResNet18**
* 📊 Metadata and **quality score** generation
* 🗌️ Visualisation with **Streamlit interactive map**

---

## 📁 Project Structure

```
marking-quality-assessment/
│
├── images/                         # Raw images (from Mapillary)
├── output/                         # Enhanced images after preprocessing
├── masks/                          # Segmentation masks (SAM output)
├── crops/                          # Cropped road markings & patches
├── datasets/                       # For classification training
├── crops_metadata.csv              # Metadata including quality score
├── geo_output.geojson              # Geo-tagged marking results
├── scripts/
│   ├── mapillary_downloader.py
│   ├── enhancement_manual.py
│   ├── Run_YOLOv8_Inference.py
│   ├── all_crops_and_condition_classification.py
│   ├── crops_metadata_with_coordinates.py
│   ├── streamlit_visuals.py
│   └── automatic_enhancement_correction.py (in progress)
```

---

## Notes

- This application is designed to assess **dashed lane markings only**.
- The **quality score** is computed based on classifier confidence, mask coverage, and YOLO detection confidence to provide a comprehensive quality indicator.
- Image enhancement is currently **manual**; automation is under development.

---

## 🔧 Instructions to Use

### 1. 🌍 Download Mapillary Images

Use the Mapillary API to download images for a region of interest:

```bash
python mapillary_downloader.py
```

Update your **latitude/longitude bounding box** and replace:

```
MLY|Your_mapillary_access_token
```

### 2. ✏️ Manual Perspective Correction

Run the tool and select the four corners of the road area:

```bash
python enhancement_manual.py
```

> *(For automated correction, see **`automatic_enhancement_correction.py`** — still in development)*

### 3. 🎯 Detect Markings (YOLOv8)

Run YOLOv8 inference to detect dashed lane markings:

```bash
python Run_YOLOv8_Inference.py
```



### 4. ✂️ Crop & Classify Markings

Run the full pipeline: generate crops (from masks) and classify them:

```bash
python all_crops_and_condition_classification.py
```

### 5. 📍 Add Coordinates to Metadata

Attach geo-coordinates to the crop metadata:

```bash
python crops_metadata_with_coordinates.py
```

### 6. 📊 View Interactive Map

Launch the Streamlit app to visualise results on a map:

```bash
streamlit run streamlit_visuals.py
```

---

## 📈 Quality Score Formula

Each marking is given a **quality score** (0 to 1) based on:

* **Coverage** (size of mask in bounding box)
* **YOLO confidence**
* **Classifier confidence**
* **Condition** (Good > Damaged > Missing)

**Formula:**

```python
quality_score = 0.4 * coverage_norm \
              + 0.3 * yolo_confidence_norm \
              + 0.3 * classifier_confidence_norm * condition_weight
```

**Condition weights:**

* Good: 1.0
* Damaged: 0.5
* Missing: 0.0

---

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```

Common libraries used:

* `opencv-python`, `numpy`, `pandas`
* `torch`, `ultralytics`, `segment-anything`
* `streamlit`, `geopandas`

---

## 🧪 Future Work

* ➕ Expansion to assess additional pavement marking types such as continuous lines, zebra crossings, and directional arrows
* ✅ Automated perspective correction (`automatic_enhancement_correction.py`)
* 🚐 Integration with real-time dashcam data
* 📍 Pavement type-specific tuning (e.g., asphalt vs concrete)

---

## 📄 License

This project is for academic and research purposes. Please cite appropriately when used in publications.

---

## 👤 Author

Developed by \[Bikesh Suwal] as part of an learning and initiative to improve road safety infrastructure using AI and computer vision.
