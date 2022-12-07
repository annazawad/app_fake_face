from flask import render_template, request
import os
import cv2
from app.fake_recognition import image_pred
import matplotlib.pyplot as matim


#global
IMAGE_FOLDER = 'static/upload'

#def index():
    #return render_template('index.html')

def app():
    return render_template('app.html')

def fake_app():
    if request.method == 'POST':
        f = request.files['image_name']
        filename = f.filename
        path = os.path.join(IMAGE_FOLDER, filename)
        f.save(path)


        if image_pred(path)==None:
            return render_template('fake2.html', noimage=True,fileupload=True)
        else:
            pred_image, predictions = image_pred(path)
            pred_filename = 'predicted_image.jpg'
            cv2.imwrite(f'./static/predict/{pred_filename}',pred_image)
            #cv2.imwrite(f'./static/predict/gray.jpg', gray)
            with open('file.txt', 'w') as data:
                data.write(str(predictions))


            gray_image = predictions[0]['crop']
            img_class = predictions[0]['prediction']
            proba = predictions[0]['probability']
            matim.imsave(f'./static/predict/gray2.jpg',gray_image, cmap='gray')
            gray = "static/predict/gray.jpg"



            return render_template('fake2.html', noimage=False, fileupload=True, img_class=img_class,proba=proba) #get request

    return render_template('fake2.html', fileupload=False) #get request