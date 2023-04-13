import tensorflow as tf
from tensorflow.keras.applications.resnet50 import decode_predictions
from tensorflow.keras.applications.resnet50 import preprocess_input
from flask import *
import numpy as np
import os
import cv2


app = Flask(__name__)

model = tf.keras.applications.ResNet50(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation="softmax",
)

@app.route("/")
def renderTemp():
    model.compile()
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def testFunction():
    if request.files['file'].filename == '':
       return jsonify({'message':"file is required",})
    file = request.files['file']
    file.save(f"uploads/{file.filename}")
    image = cv2.imread(f"uploads/{file.filename}")
    img_resized = cv2.resize(image,(224,224))
    img_arr = np.asarray(img_resized)
    img_arr = np.expand_dims(img_arr, axis = 0)
    transformedImage = preprocess_input(img_arr)
    pred = model.predict(transformedImage)
    predictionLabel = decode_predictions(pred, top = 1)
    return jsonify({
        'result':predictionLabel[0][0][1],
    })

app.run(host='0.0.0.0', port=80, debug=True)