from io import BytesIO

import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile

from utils import preprocess_image, prettify_result


model_handle = "https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_640x640/1"

detector = hub.load(model_handle).signatures["serving_default"]


app = FastAPI()


@app.get("/")
def home():
    return {
        "health_check": "OK",
        "model_version": "0.1.0",
        }


@app.post("/api/predict")
def create_upload_file(file: UploadFile = File(...)):

    img_tensor = preprocess_image(file.file)
    result = detector(img_tensor)
    detections_dict = prettify_result(result)
    
    return detections_dict


