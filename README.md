# Explainable-AI-based-Fake-health-news-Detection
## 📌 Project Overview
This project builds a robust, end-to-end Machine Learning pipeline to detect health-related misinformation. Because fake news strategies constantly evolve, a model trained on general health claims might fail when applied to niche outbreaks (like COVID-19). 

To solve this, this project ingests and standardizes three distinct datasets—**PubHealth**, **CoAID**, and **FakeHealth**—into a unified binary classification format (REAL vs. FAKE). By doing so, it allows for the training of multiple machine learning model and rigorously tests their generalizability through **cross-dataset evaluation**.

## 🎯 What We Wanted to Do (Project Goals)
1. **Unified Data Pipeline:** Programmatically fetch, merge, and clean disparate health-news datasets with varying structures and label taxonomies into a standard `(combined_text, label_binary)` format.
2. **Comprehensive ML Training:** Contrast the performance of lightweight traditional models (Multinomial Naive Bayes, Linear SVC) with high-performance ensemble trees (XGBoost, CatBoost).
3. **Cross-Dataset Generalizability:** Prove that the models are learning semantic linguistic patterns of misinformation, rather than just memorizing vocabulary specific to a single dataset (e.g., training on PubHealth and testing on FakeHealth).
4. **Explainable AI (XAI):** Utilize **LIME** to demystify the "black box" of the models, ensuring that predictions are based on logical indicators of deception rather than spurious dataset artifacts.

## Dataset Standardization Breakdown
The preprocessing pipeline successfully aggregated the datasets into binary classifications:
* **PubHealth:** 10,701 clean rows (53% REAL, 47% FAKE)
* **FakeHealth:** 2,296 clean rows (67% REAL, 33% FAKE)
* **CoAID:** 2,133 clean rows (73% REAL, 27% FAKE)
![Dataset Details](https://github.com/Gourab-Chowdhury/Explainable-AI-based-Fake-health-news-Detection/blob/main/Images/EDA/Dataset%20Size%20and%20Label%20Comparision.png)

## Dataset Model Performance
### PubHealth Results
![PubHealth Results](https://github.com/Gourab-Chowdhury/Explainable-AI-based-Fake-health-news-Detection/blob/main/Images/Performance%20and%20Evalution/Perfomance%20Matrix%20-%20PubHealth.png)

### FakeHealth Results
![FakeHealth Results](https://github.com/Gourab-Chowdhury/Explainable-AI-based-Fake-health-news-Detection/blob/main/Images/Performance%20and%20Evalution/Perfomance%20Matrix%20-%20FakeHealth.png)

### CoAid Results
![CoAid Results](https://github.com/Gourab-Chowdhury/Explainable-AI-based-Fake-health-news-Detection/blob/main/Images/Performance%20and%20Evalution/Perfomance%20Matrix%20-%20CoAid.png)

### 📊 Result Analysis
![Results](https://github.com/Gourab-Chowdhury/Explainable-AI-based-Fake-health-news-Detection/blob/main/Images/Performance%20and%20Evalution/Cross-data%20generalization.png)




## 4. Explainability Insights (SHAP/LIME):
LIME text explainers consistently highlighted phrases related to verifiable sources (e.g., "according to the CDC", "published in the journal of...").
### Naive Bayes
![Naive Bayes](https://github.com/Gourab-Chowdhury/Explainable-AI-based-Fake-health-news-Detection/blob/main/Images/Explanable%20Ai/xAi%20-%20Naive%20Bayes.png)

### Logistic Regrassion
![Logistic REgression](https://github.com/Gourab-Chowdhury/Explainable-AI-based-Fake-health-news-Detection/blob/main/Images/Explanable%20Ai/xAi%20-%20Logistic%20Regression.png)

### Support Vector Machine
![SVM](https://github.com/Gourab-Chowdhury/Explainable-AI-based-Fake-health-news-Detection/blob/main/Images/Explanable%20Ai/XAi%20-%20SVM.png)

### catboost
![catboost](https://github.com/Gourab-Chowdhury/Explainable-AI-based-Fake-health-news-Detection/blob/main/Images/Explanable%20Ai/xAi%20catboost.png)

### LightGBM
![LightGBM](https://github.com/Gourab-Chowdhury/Explainable-AI-based-Fake-health-news-Detection/blob/main/Images/Explanable%20Ai/xAi%20-%20LightGBM.png)

### XGBoost
![XGBoost](https://github.com/Gourab-Chowdhury/Explainable-AI-based-Fake-health-news-Detection/blob/main/Images/Explanable%20Ai/xAi%20-%20XGBoost.png)
