import subprocess
subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
import streamlit as st
import numpy as np
import nibabel as nib
import tensorflow as tf
import pickle
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.image import resize
import tempfile
import os
import time
import matplotlib.pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

model_path = "ensemble_model_B1.pkl"
scaler_path = "scaler_B1.pkl"

with open(model_path, "rb") as model_file:
    ensemble_clf = pickle.load(model_file)
with open(scaler_path, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

def get_resnet():
    base_model = ResNet50(weights="imagenet", include_top=False)
    return Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))

def get_effnet():
    base_model = EfficientNetB0(weights="imagenet", include_top=False)
    return Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))

resnet_model = get_resnet()
effnet_model = get_effnet()

def preprocess_nii(nii_path):
    img = nib.load(nii_path).get_fdata()
    img = np.clip(img, 0, np.percentile(img, 99))  # Normalize intensity
    img = (img - img.min()) / (img.max() - img.min())  # Scale to 0-1
    img = np.stack((img,) * 3, axis=-1)  # Convert grayscale to 3-channel
    img = resize(img, (224, 224)).numpy()
    img = img.reshape(-1, 224, 224, 3)  # Reshape for CNN
    return img

def extract_features(img, model):
    features = model.predict(img, verbose=0, batch_size=1)
    return np.mean(features, axis=0)

def predict_tumor(nii_path):
    img = preprocess_nii(nii_path)
    resnet_features = extract_features(img, resnet_model)
    effnet_features = extract_features(img, effnet_model)
    fused_features = np.concatenate([resnet_features, effnet_features])
    fused_features = scaler.transform([fused_features])
    prediction = ensemble_clf.predict(fused_features)[0]
    return "HGG (High-Grade Glioma)" if prediction == 1 else "LGG (Low-Grade Glioma)"

st.set_page_config(page_title="Glioma Grading", page_icon="🧠", layout="centered")

selected_tab = st.selectbox("Navigation", ["Home", "MRI Classification"])

if selected_tab == "Home":
    st.title("🧠 Understanding Glioma")
    st.write("""
    **Glioma** is a type of tumor that occurs in the brain and spinal cord. It originates in the glial cells, which
    support nerve functions. Gliomas can be **low-grade (LGG)** or **high-grade (HGG)**, affecting prognosis and treatment.
    
    ### 🔍 Symptoms:
    - Headaches, nausea, and seizures
    - Memory loss or cognitive difficulties
    - Blurred vision or speech problems
    
    ### 🎯 Causes & Risk Factors:
    - Genetic mutations & hereditary links
    - Exposure to radiation
    - Age-related risks
    
    🏥 **Early diagnosis** through MRI scans is crucial for effective treatment.
    """)

elif selected_tab == "MRI Classification":
    st.title("🧠 MRI Tumor Classification")
    uploaded_file = st.file_uploader("Upload an MRI scan (.nii file)", type=["nii", "nii.gz"])
    
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nii") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name
        
        img = nib.load(temp_path).get_fdata()
        middle_slice = img[:, :, img.shape[2] // 2]
        fig, ax = plt.subplots()
        ax.imshow(middle_slice, cmap='gray')
        ax.axis('off')
        st.pyplot(fig)
        
        st.write("✅ File Uploaded Successfully!")
        
        with st.spinner('🔄 Analyzing MRI Scan...'): 
            steps = [
                "📡 Preprocessing MRI Image...",
                "🔍 Extracting features using ResNet50...",
                "🧠 Extracting features using EfficientNet...",
                "🔗 Combining extracted features...",
                "📊 Running classification model..."
            ]
            for step in steps:
                st.write(step)
                time.sleep(1) 
            
            prediction = predict_tumor(temp_path)
            os.remove(temp_path)
        
        color = "#ff4d4d" if "HGG" in prediction else "#4CAF50"
        st.markdown(
            f'<div style="text-align: center; padding: 20px; border-radius: 10px; background-color: {color}; color: white; font-size: 24px;">'
            f'🎯 <b>Predicted Tumor Type:</b> {prediction}</div>',
            unsafe_allow_html=True
        )

        st.write("### 🏥 Preventive Measures & Recommendations")
        
        if "HGG" in prediction:
            st.error("⚠️ **High-Grade Glioma (HGG) detected.** It is aggressive and requires immediate attention.")
            st.write("""
            - 🏥 **Consult an oncologist** for treatment options such as **surgery, chemotherapy, or radiation therapy.**
            - 🍏 Maintain a **healthy diet** rich in antioxidants, vitamins, and fiber.
            - 💪 Engage in **moderate exercise** to improve strength and immunity.
            - 🧘 Reduce stress through **meditation and relaxation techniques.**
            - 🩺 Regular **follow-ups and MRI scans** to monitor tumor progression.
            """)

        else:
            st.success("✅ **Low-Grade Glioma (LGG) detected.** It grows slowly but requires monitoring.")
            st.write("""
            - 🩺 **Regular MRI scans** to check for any progression to a high-grade glioma.
            - 🍎 A **balanced diet** with low sugar intake can help overall health.
            - 🏋️ Stay **physically active** to maintain body strength.
            - 🚭 Avoid exposure to **harmful chemicals and radiation.**
            - 🧠 **Cognitive therapy** may help manage potential symptoms.
            """)

