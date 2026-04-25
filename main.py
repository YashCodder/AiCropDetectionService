from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np
from PIL import Image
import io
import tensorflow as tf

app = FastAPI()

# ==============================
# LOAD TRAINED MODEL
# ==============================
model = tf.keras.models.load_model("model.h5")

# Class labels (must match training order)
classes = ["Healthy", "Yellow Fungus", "Other Disease"]

# ==============================
# SOIL DATA MODEL (keep as is)
# ==============================
class SoilData(BaseModel):
    temperature: float
    humidity: float
    ph: float

# ==============================
# HOME ROUTE
# ==============================
@app.get("/")
def home():
    return {"message": "AI Crop Disease API is running"}

# ==============================
# SOIL PREDICTION (keep as placeholder for now)
# ==============================
@app.post("/predict-soil")
def predict_soil(data: SoilData):
    risk = (data.humidity * 0.4 + data.temperature * 0.3 + (7 - abs(data.ph - 7)) * 10) % 100
    return {
        "risk_percentage": round(risk, 2),
        "message": "High Risk" if risk > 60 else "Moderate Risk"
    }

# ==============================
# IMAGE PREDICTION (REAL MODEL)
# ==============================
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()

    # Read and preprocess image
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((224, 224))

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    confidence = float(np.max(prediction)) * 100

    return {
        "prediction": classes[class_index],
        "confidence": round(confidence, 2)
    }