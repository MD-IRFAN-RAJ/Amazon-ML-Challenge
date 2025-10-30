# ML Challenge 2025 -- Smart Product Pricing

**Team Name:** Lower_Bound
**Member:** Md Irfan Raj, Abishek E (c) , Raushan Kumar

------------------------------------------------------------------------

## 🧠 Methodology Used

We approached this as a **supervised regression problem**, predicting
product prices based on catalog text and image data.\
Our goal was to minimize **SMAPE (Symmetric Mean Absolute Percentage
Error)** and **MAE (Mean Absolute Error)**.

------------------------------------------------------------------------

## 🧩 Model Architecture / Algorithms

-   **Text Representation:** TF-IDF (Term Frequency--Inverse Document
    Frequency) with bi- and tri-grams\
-   **Regressor:** Ridge Regression (L2-regularized linear regression)\
-   **Scaling:** StandardScaler for stabilizing target variable
    distribution\
-   **Vector Size:** 30,000 text features

------------------------------------------------------------------------

## ⚙️ Feature Engineering Techniques

-   Cleaned and normalized product descriptions (lowercasing, removing
    punctuation, extra spaces)
-   Used TF-IDF for catalog_content to capture key textual signals like
    brand names, sizes, weights, and product variants.
-   Standardized target (`price`) for smoother optimization
-   Applied outlier clipping for predicted prices within a reasonable
    range (0--1000)
-   (Optional extension: simple image features such as mean RGB color or
    brightness can be fused later)

------------------------------------------------------------------------

## 📊 Evaluation

-   **Validation MAE:** ≈ 14.2110\
-   **Validation SMAPE:** ≈ 54 %\
-   **Dataset Split:** 85% training, 15% validation

------------------------------------------------------------------------

## 🚀 Final Output

-   Output file: `test_out.csv`
-   Columns: `sample_id`, `price`
-   Same row count as `test.csv` (no missing or extra samples)
