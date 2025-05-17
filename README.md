<h1 align="center">🧠 Glioma Grading using Ensemble Deep Learning</h1>
<p align="center">
  <strong>Multi-Modal Feature Fusion with ResNet50 & EfficientNet Variants</strong><br>
  <em>Classifying Brain Tumors into High-Grade Glioma (HGG) and Low-Grade Glioma (LGG)</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Made%20With-Python%203.10-blue?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Framework-Streamlit-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Dataset-BraTS%202018-brightgreen?style=for-the-badge" />
</p>

---

## 📘 Project Overview

This project presents an AI-based web application to classify brain tumors as **Low-Grade Glioma (LGG)** or **High-Grade Glioma (HGG)** from MRI scans. The system uses **ensemble deep learning models (ResNet50 + EfficientNetB0)** for feature extraction and a stacked ensemble of traditional machine learning classifiers for final predictions.

> 🎯 Achieved **98.0% classification accuracy** using XGBoost on fused features.

---

## 📊 Methodology

### 🔄 Workflow

1. Upload NIfTI MRI scan
2. Preprocess (Normalize → Resize → Convert to RGB)
3. Extract features using ResNet50 and EfficientNet
4. Fuse features (mean+variance)
5. Classify using trained ensemble model
6. Return prediction and confidence

🧠 Architecture
CNNs Used: ResNet50, EfficientNetB0–B4

Ensemble Classifiers: XGBoost, SVM, Random Forest, Logistic Regression

Evaluation Metrics: Accuracy, Precision, Recall, F1-score

🛠️ Technologies Used
Category	Tools / Frameworks
Programming	Python 3.10
Deep Learning	TensorFlow, Keras
Machine Learning	Scikit-learn, XGBoost
Web Interface	Streamlit
Visualization	Matplotlib, Seaborn
Image Processing	OpenCV, NiBabel
IDE	VS Code
Dataset	MICCAI BraTS 2018 (LGG vs HGG)

📷 Screenshots
🏠 Home Page
<img src="images/homepage.png" width="800"/>
🔍 Upload & Classify
<img src="images/upload-classify.png" width="800"/>
✅ Prediction Output
<img src="images/prediction-hgg.png" width="800"/>
🚀 How to Run the App Locally
bash
Copy
Edit
# Step 1: Clone the Repository
git clone https://github.com/yourusername/glioma-grading-app.git
cd glioma-grading-app

# Step 2: Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Run the Streamlit app
streamlit run app.py
🧪 Sample Prediction
Upload .nii or .nii.gz format MRI scan files.
Output includes tumor grade (HGG/LGG) + confidence scores.


<p align="center"><strong>Made with ❤️ for clinical AI research</strong></p>
