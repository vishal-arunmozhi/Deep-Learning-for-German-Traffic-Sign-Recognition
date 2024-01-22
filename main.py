from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List

import imageio
import numpy as np
import keras
import cv2

app = FastAPI()

model = keras.models.load_model("gtsrb_model.h5")

# function takes as input a list of uploaded image files and returns a list of processed images and a list of file names
def process_images(images):
    processed_images = []
    file_names = []

    for image in images:
        file_names.append(image.filename)

        # reading .ppm files into numpy array
        img = imageio.imread(image.file) 

        # changing the default 'RGB' format to 'BGR' format since that's the format the model was trained on
        image_bgr = img[:, :, ::-1] 

        # resizing the image to 32x32
        resized_img = cv2.resize(image_bgr, (32, 32)) 

        normalized_img = resized_img/255.0
        processed_images.append(normalized_img)
    
    processed_images = np.array(processed_images)

    return processed_images, file_names


@app.post("/predict")
def predict(images: List[UploadFile] = File(...)):

    processed_images, file_names = process_images(images)

    predictions = model.predict(processed_images, batch_size=256)

    class_pred = np.argmax(predictions, axis=1).tolist()
    prob_scores = np.max(predictions, axis=1).tolist()
    confidence_scores = [round(score, 2) for score in prob_scores]

    response = {}
    for i, file_name in enumerate(file_names):
        response[file_name] = {"class": class_pred[i], "confidence":confidence_scores[i]}

    return JSONResponse(content=response)