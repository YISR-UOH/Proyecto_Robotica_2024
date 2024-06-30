from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from ultralytics import YOLO,RTDETR
import json

app = FastAPI()

model = YOLO("ProyectoFinal/train/Yolov8n/weights/Yolov8n.pt")


@app.get("/")
async def read_root():
 return {"message": "Hello, World!"}

@app.post("/detect/")
async def detect_objects(file: UploadFile):
  # Process the uploaded image for object detection
  image_bytes = await file.read()
  image = np.frombuffer(image_bytes, dtype=np.uint8)
  image = cv2.imdecode(image, cv2.IMREAD_COLOR)
  
  detections = model.predict(image, save=False)
  detections = detections[0].tojson()
  return {"detections": detections}

# python -m uvicorn main:app --reload