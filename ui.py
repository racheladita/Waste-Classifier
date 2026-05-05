import streamlit as st
import cv2
import numpy as np
import joblib
import os
from scipy.cluster.vq import vq
from PIL import Image

# Page config
st.set_page_config(
    page_title="Waste Classifier AI",
    page_icon="♻️",
    layout="wide"
)

# Title and Description
st.title("♻️ Automated Waste Classifier")
st.markdown("""
This application uses **Computer Vision (SIFT)** and **Machine Learning (SVM)** to classify waste into six categories.
It demonstrates a classical **Bag of Visual Words (BoVW)** pipeline.
""")

# Load model function
@st.cache_resource
def load_model():
    model_path = "trashnet.pkl"
    if not os.path.exists(model_path):
        return None, None, None, None
    return joblib.load(model_path)

clf, classes_names, k, voc = load_model()

# Sidebar
st.sidebar.header("About the Model")
st.sidebar.info("""
- **Algorithm:** SIFT + SVM (RBF Kernel)
- **Vocabulary Size:** 250 Visual Words
- **Training Accuracy:** 88%
- **Test Accuracy:** 63%
""")


if clf is None:
    st.error("Model file `trashnet.pkl` not found. Please ensure it is in the project root.")
else:
    # File Uploader
    uploaded_file = st.file_uploader("Choose an image of waste...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Layout columns
        col1, col2 = st.columns(2)

        # Process and Display Image
        image = Image.open(uploaded_file)
        img_array = np.array(image)
        
        with col1:
            st.image(image, caption="Uploaded Image", width="stretch")

        with st.spinner('Analyzing features...'):
            # Prepare for CV2
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array

            # SIFT Extraction
            sift = cv2.SIFT_create()
            kpts, des = sift.detectAndCompute(gray, None)

            # Feature Quantization
            test_features = np.zeros((1, k), "float32")
            if des is not None:
                words, distance = vq(des, voc)
                for w in words:
                    test_features[0][w] += 1
                
                # Prediction
                prediction_idx = clf.predict(test_features)[0]
                predicted_class = classes_names[prediction_idx]

                with col2:
                    st.subheader("Classification Result")
                    st.success(f"**Identified Category:** {predicted_class}")
                    
                    # Confidence/Quality Indicator
                    num_kpts = len(kpts)
                    if num_kpts < 100:
                        st.warning(f"⚠️ **Low Feature Density** ({num_kpts} points). The image may be too plain or blurry for a highly accurate classification.")
                    elif num_kpts < 500:
                        st.info(f"ℹ️ **Medium Feature Density** ({num_kpts} points). Sufficient features detected for analysis.")
                    else:
                        st.success(f"✅ **High Feature Density** ({num_kpts} points). Rich visual features detected.")
                    
                    # Technical details
                    st.markdown("---")
                    st.write(f"**Interest Points (SIFT) Detected:** {len(kpts)}")
                    
                    # Show Histogram
                    st.write("**Visual Word Distribution:**")
                    st.bar_chart(test_features[0])
            else:
                st.error("Could not find enough visual features in this image. Try a clearer photo.")

st.markdown("---")
st.caption("Developed by Adita Putri Puspaningrum as part of a Waste Classification Research Project.")
