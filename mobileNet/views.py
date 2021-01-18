import os
from django.shortcuts import render
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
import numpy as np
import urllib.request

# Create your views here.
def index(request):
    return render(request, 'index.html')

def local_image(request):
    image_file = request.FILES['image_file']
    print(image_file)
    with open('static/static-image.jpg', 'wb') as f:
        for chunk in image_file.chunks():
            f.write(chunk)

    prediction = predict_image('static/static-image.jpg')

    return render(request, 'index.html', {'prediction': prediction, "local_image_file": image_file})

def web_image(request):
    image_file = request.POST['image_file']

    urllib.request.urlretrieve(image_file, "static/static-image.jpg")

    prediction = predict_image('static/static-image.jpg')

    return render(request, 'index.html', {'prediction': prediction, "web_image_file": image_file})

def predict_image(file):
    # run on cpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    vgg16 = tf.keras.applications.vgg16.VGG16()

    img = image.load_img(file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)

    preprocessed_image = tf.keras.applications.vgg16.preprocess_input(img_array_expanded_dims)
    predictions = vgg16.predict(preprocessed_image)

    return imagenet_utils.decode_predictions(predictions)[0]
