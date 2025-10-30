# 🧠 Amazon ML Challenge 2024 – Approach Report  
**Author:** Md Irfan Raj  
**Members** : Md Irfan Raj , Abishek E, Raushan Kumar
**Challenge:** Amazon ML Challenge 2024  
**Metric:** Symmetric Mean Absolute Percentage Error (SMAPE)  
**Final Score:** ~54%

---

## 📌 Overview
This repository documents my approach and implementation for the **Amazon ML Challenge 2024**, an applied machine learning competition hosted by Amazon.  
The task involved developing a model capable of predicting key values from complex multimodal datasets (image + metadata) to minimize the **SMAPE metric**.

---

## ⚙️ Problem Understanding
The dataset provided contained both **image URLs** and **metadata (CSV format)**.  
Each image corresponded to a specific product or listing with associated attributes.  
The goal was to predict the target variable (hidden by the organizers) as accurately as possible.

---

## 🧩 Data Preprocessing

### 1. Image Download & Handling
- Implemented a **multi-threaded image downloader** using `ThreadPoolExecutor` to efficiently handle thousands of images.
- Created **placeholder gray images** for URLs that failed to download after multiple retries, ensuring consistent dataset size.
- Used the **PIL (Python Imaging Library)** for image creation and validation.
- Verified dataset integrity by cross-checking all filenames between `train.csv` and the downloaded folder.

```python
with ThreadPoolExecutor(max_workers=60) as executor:
    futures = [executor.submit(download_one, url) for url in image_links]
```

### 2. Data Validation
- Ensured **1:1 correspondence** between CSV entries and downloaded images.
- Logged failed downloads in a text file for auditability.

---

## 🧠 Model Development

### 1. Feature Engineering
- Extracted **image embeddings** (if applicable) and combined them with **tabular features** for hybrid modeling.
- Normalized numeric features using **MinMaxScaler**.
- Encoded categorical variables with **LabelEncoder** / One-Hot encoding as appropriate.

### 2. Model Selection
Experimented with:
- **Gradient Boosting Models (XGBoost / LightGBM)**
- **Random Forest Regressor**
- **Linear Regression Baseline**

LightGBM provided the most stable results on validation data.

---

## 📊 Evaluation

### Metric Used:
**Symmetric Mean Absolute Percentage Error (SMAPE)**  
\[
SMAPE = \frac{100\%}{N} \sum_{i=1}^{N} \frac{|F_i - A_i|}{(|A_i| + |F_i|)/2}
\]

### Final Result:
| Metric | Value |
|:-------:|:------:|
| **SMAPE** | **~54%** |

Achieving 54% SMAPE placed the model in a competitive range among submissions, considering limited training epochs and resource constraints.

---

## 🚀 Key Insights
- **Parallelization** of image downloads saved hours of preprocessing time.  
- **Placeholder creation** ensured no training interruption due to missing files.  
- Data normalization and image-text feature fusion improved model robustness.  
- Further performance could be achieved through **deep CNN fine-tuning** (e.g., ResNet50 + metadata fusion).

---

## 💡 Future Work
- Implement end-to-end **Vision Transformer (ViT)** or **CLIP-based embeddings** for richer visual understanding.  
- Add **data augmentation** to increase generalization.  
- Experiment with **weighted SMAPE loss** for improved metric alignment.  
- Deploy inference pipeline for faster testing and validation.

---

## 🧾 Tech Stack
| Category | Tools/Packages |
|-----------|----------------|
| Data Handling | Pandas, NumPy |
| Parallelization | ThreadPoolExecutor |
| Image Processing | PIL |
| Model Training | XGBoost, LightGBM, Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| Evaluation | Custom SMAPE metric function |

---

## 🏁 Conclusion
This project was a valuable learning experience in handling large-scale data pipelines, optimizing download performance, and integrating image + metadata models.  
Despite challenges with missing images and limited compute, the final SMAPE of **~54%** demonstrated a strong baseline model for the competition task.

---

**Author:** [Md Irfan Raj](https://www.linkedin.com/in/md-irfan-raj)  
**Institute:** Indian Institute of Information Technology, Bhagalpur  
