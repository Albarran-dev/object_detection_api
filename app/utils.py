from io import BytesIO

import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from PIL import Image


INPUT_SHAPE = (640, 640)
IDX_TO_CLASS = {
    1: 'person',
    3: 'car',
}

def preprocess_image(file, input_shape=INPUT_SHAPE):
    pil_image = Image.open(BytesIO(file.read()))
    pil_image = pil_image.resize(input_shape)
    img = np.array(pil_image)
    img = np.expand_dims(img, 0)
    img_tensor = tf.convert_to_tensor(img, tf.uint8)
    return img_tensor


def prettify_result(result, idx_to_class=IDX_TO_CLASS):
    """
    Returns result data from tf_model in the following format:
    'i': {
        "object_detected": 'car' or 'person'
        "confidence_score": 0-1
        "bounding_box": [ymin, xmin, ymax, xmax]
    }
    The detections are ordered in descending order by confidence score.
    """
    result = {key:value.numpy() for key,value in result.items()}
        
    detection_idx = result["detection_classes"][0]
    detection_boxes = result["detection_boxes"][0]
    detection_scores = result["detection_scores"][0]
        
    detections = []
    for idx_class, detection_box, detection_score in zip(detection_idx, 
                                                         detection_boxes, 
                                                         detection_scores):
        if not idx_class in (1, 3):
            continue
        
        detection = (idx_to_class[idx_class],
                    float(detection_score),
                    detection_box.astype(float).tolist())        
        detections.append(detection)
        
    detections.sort(key=lambda x: x[1], reverse=True)
    detections_dict = {i: {"object_detected": detection[0],
                           "confidence_score": detection[1],
                           "bounding_box": detection[2]} 
                       for i, detection in enumerate(detections)}
    return detections_dict