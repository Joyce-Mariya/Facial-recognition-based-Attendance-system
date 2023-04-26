from fastapi import FastAPI
from fastapi import File, UploadFile

import numpy as np
import cv2
from keras.models import load_model

from datetime import datetime

model = load_model('keras_model.h5')

app = FastAPI()
# @app.get("/")
# async def index():
#    return {"message": "Hello World"}

@app.post("/upload")
def upload(file: UploadFile = File(...)):
    name = ''
    try:
        contents = file.file.read()
        with open('input.jpg', 'wb') as f:
            f.write(contents)
        name = findPerson()
    except Exception:
        return {"message": "Something went wrong", "status": "failed"}
    finally:
        file.file.close()
    now = datetime.now()
    dt = now.strftime("%d/%m/%Y %H:%M:%S")

    return {"message": {"name": name, "Date-Time": dt, "lat":----, "long": ----}, "status": "successful"}

def findPerson():
    img = cv2.imread('./input.jpg')
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (img / 127.5) - 1

    # Predicts the model
    prediction = model.predict(image)
    index = np.argmax(prediction)
    confidence_score = prediction[0][index]

    print(get_className(index), confidence_score)
    return get_className(index)

def get_className(classNo):
    if classNo == 0:
        return "name1"
    elif classNo == 1:
        return "name2"
