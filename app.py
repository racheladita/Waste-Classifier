import os
import io
import cv2
import numpy as np
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from scipy.cluster.vq import vq

# Initialize FastAPI app
app = FastAPI(
    title="Waste Classifier API",
    description="Production-ready inference server for classifying waste into 6 categories using SIFT + SVM.",
    version="1.0.0"
)

# Global variables to store the model and vocabulary
clf = None
classes_names = None
k = None
voc = None

@app.on_event("startup")
def load_model():
    """Load the trained model and visual vocabulary on startup."""
    global clf, classes_names, k, voc
    model_path = "trashnet.pkl"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found. Please run the training notebook first.")
    
    # Load the serialized objects
    clf, classes_names, k, voc = joblib.load(model_path)
    print("Model and vocabulary loaded successfully.")

@app.get("/health")
def health_check():
    """Standard health check endpoint."""
    return {"status": "healthy", "model_loaded": clf is not None}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept an image file and return the predicted waste category.
    """
    if clf is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    # Read image bytes
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kpts, des = sift.detectAndCompute(gray, None)

    # Feature Quantization (Bag of Visual Words)
    test_features = np.zeros((1, k), "float32")
    if des is not None:
        words, distance = vq(des, voc)
        for w in words:
            test_features[0][w] += 1
    else:
        # No keypoints found
        pass

    # Prediction
    prediction_idx = clf.predict(test_features)[0]
    predicted_class = classes_names[prediction_idx]
    
    # Get probability/confidence if available (SVC usually needs probability=True during training)
    # Since the current model was trained with default SVC, we just return the label.
    
    return {
        "filename": file.filename,
        "prediction": predicted_class,
        "class_index": int(prediction_idx)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
