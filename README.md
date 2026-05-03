# Waste Classification via Bag of Visual Words (SIFT + SVM)

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.10+-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)

An automated waste classification system implementing Computer Vision and Machine Learning techniques to facilitate recycling efficiency. This project utilizes the Scale-Invariant Feature Transform (SIFT) for feature extraction and Support Vector Machines (SVM) for classification.

## 📊 System Pipeline

```mermaid
graph TD
    subgraph Training Phase
        T1[(TrashNet Dataset)] --> |85% Split| T2[Training Images]
        T2 --> T3[SIFT Feature Extraction]
        T3 --> |Millions of Descriptors| T4[K-Means Clustering]
        T4 --> |250 Words| T5[Visual Vocabulary]
        T3 --> T6[Vector Quantization]
        T5 --> T6
        T6 --> |Histograms| T7[Train SVM Classifier]
        T7 --> |trashnet.pkl| T8[(Serialized Model)]
    end

    subgraph Inference Phase
        I1[New Image] --> I2[SIFT Feature Extraction]
        I2 --> I3[Vector Quantization]
        T5 -.-> |Load Vocabulary| I3
        I3 --> |Histogram| I4[Predict with SVM]
        T8 -.-> |Load Model| I4
        I4 --> I5([Predicted Class: Glass, Metal, etc.])
    end

    style T8 fill:#f96,stroke:#333,stroke-width:2px,color:#000
    style I5 fill:#9f9,stroke:#333,stroke-width:2px,color:#000
```

## 📖 Project Overview

The objective is to classify solid waste into six categories: **Cardboard, Glass, Metal, Paper, Plastic, and Trash**. The implementation follows a **Bag of Visual Words (BoVW)** pipeline, providing a transparent approach to feature quantification and classification compared to opaque end-to-end models.

## 🛠️ Methodology

The system follows a classical Computer Vision pipeline:

1.  **Feature Extraction (SIFT)**: Detects local interest points and extracts descriptors invariant to scale, rotation, and illumination.
2.  **Vocabulary Building (K-Means)**: Clusters millions of extracted SIFT descriptors into a visual vocabulary of 250 words.
3.  **Vector Quantization**: Maps image features to the visual vocabulary, generating frequency histograms for each image.
4.  **Classification (SVM)**: Utilizes a Support Vector Machine with a Radial Basis Function (RBF) kernel for multi-class classification.

## 🧠 Technical Deep Dive

While modern deep learning (CNNs) dominates image classification, this project implements a **transparent pipeline** that focuses on explicit feature engineering:

- **SIFT vs. Deep Learning**: SIFT allows the model to remain effective even with smaller datasets by focusing on distinctive geometric structures (edges, corners) rather than requiring millions of parameters to learn low-level features from scratch.
- **K-Means as a Feature Compressor**: By clustering descriptors into a 250-word vocabulary, the system effectively compresses high-dimensional image data into a compact 250-dimension histogram, significantly reducing the computational cost of the final SVM classification.
- **Kernel Selection**: The **RBF Kernel** was chosen to handle the non-linear distribution of visual words in the feature space, allowing for more complex decision boundaries between similar categories (e.g., Plastic vs. Glass).

## ⚙️ Backend Integration Potential

From a backend engineering perspective, the system is designed for **portability and integration**:

- **Model Serialization**: Using `joblib`, the trained SVM classifier and the visual vocabulary (`voc`) are serialized into a single `trashnet.pkl` file. This allows the model to be loaded into a production environment (such as a FastAPI or Flask microservice) without retraining.
- **Inference Pipeline**: The inference logic is decoupled from training. To process a new request, the backend only needs to:
  1.  Perform SIFT extraction on the input image.
  2.  Quantize features using the pre-loaded vocabulary.
  3.  Run the prediction via the SVM.
- **Decoupled Architecture**: The data extraction and organizing logic are scripted to ensure that the dataset pipeline is reproducible and can be easily adapted for different data sources.

## 📊 Dataset: TrashNet

The model is trained and evaluated on the [TrashNet dataset](https://github.com/garythung/trashnet), consisting of 2,527 images of waste objects.

- **Training Set**: 85%
- **Test Set**: 15%

## 🚀 Installation & Requirements

This project is compatible with Python environments version 3.10 and above.

```bash
pip install opencv-contrib-python numpy scipy scikit-learn joblib matplotlib seaborn pandas
```

## 📈 Results

Evaluation results on the test set:

| Category      | Precision | Recall | F1-Score |
| :------------ | :-------- | :----- | :------- |
| **Paper**     | 0.766     | 0.800  | 0.783    |
| **Cardboard** | 0.667     | 0.689  | 0.677    |
| **Plastic**   | 0.640     | 0.658  | 0.649    |
| **Metal**     | 0.700     | 0.452  | 0.549    |
| **Glass**     | 0.448     | 0.618  | 0.519    |
| **Trash**     | 0.500     | 0.143  | 0.222    |

**Overall Test Accuracy: 62.7%**

## 📂 Repository Structure

- `Waste_Classifier.ipynb`: Complete pipeline from data extraction to evaluation.
- `trashnet.pkl`: Pre-trained SVM model and visual vocabulary.

---

_Developed by Adita Putri Puspaningrum._
