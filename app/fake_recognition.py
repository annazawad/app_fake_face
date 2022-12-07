import numpy as np
import sklearn
import tensorflow
from tensorflow import keras
import cv2
import math

haar = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')

# load json and create model
json_file = open('./model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = tensorflow.keras.models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./model/model.h5")
print("Loaded model from disk")
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

def image_pred(pathfile):
    try:
        img = cv2.imread(pathfile)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(gray, scaleFactor=2, minNeighbors=5)
        prediction_list = []

        for x1, y1, w, h in faces:
            crop = img[y1:y1 + h, x1:x1 + w]
            crop = crop / 255.
            crop2 = gray[y1:y1 + h, x1:x1 + w]
            width = crop.shape[0]
            if width >= 100:
                img_resized = cv2.resize(crop, (100, 100), cv2.INTER_AREA)
            else:
                img_resized = cv2.resize(crop, (100, 100), cv2.INTER_CUBIC)

            arr_img_reshaped = img_resized.reshape(1, 100, 100, 3)
            probability = round(loaded_model.predict(arr_img_reshaped)[0][0], 5)
            if probability > 0.5:
                prob = probability
                if_fake = 'fake person'
            else:
                prob = 1 - probability
                if_fake = 'real person'

            # text = f'{probability}'
            text = f'{if_fake} : {round(prob * 100, 2)}%'
            if if_fake == 'fake person':
                color = (250, 100, 100)
            else:
                color = (250, 100, 100)
            if w > 400:
                k = 4
            else:
                k = 0.7
            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + w), color, math.floor(k + 1))
            cv2.putText(img, text, (x1, y1), cv2.FONT_HERSHEY_PLAIN, k, (255, 255, 255), math.floor(k + 1))

            output = {
                'crop': crop2,
                'prediction': if_fake,
                'probability': round(prob * 100, 2),
                'prob_fake': round(loaded_model.predict(arr_img_reshaped)[0][0] * 100, 2)
            }
        prediction_list.append(output)

        return img, prediction_list

    except:
        return None

