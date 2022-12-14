from io import BytesIO

import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile

from utils import preprocess_image, prettify_result


# model_handle = "./model/ssd_mobilenet_v2_2"
model_handle = "./model/faster_rcnn_inception_resnet_v2_640x640_1"
# model_handle = "/Users/alvaro_albarran/github/turing_challenge_entry_challenge/app/model/ssd_mobilenet_v2_2"

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


