import base64
import tensorflow as tf
from skimage import transform
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from PIL import Image
model=tf.keras.models.load_model('version_1.h5')

app=Flask(__name__,template_folder='.')
@app.route("/")
def index_view():
    return render_template("index.html")
@app.route("/predict",methods=['POST','GET'])
def predict_model():
    if request.method=='POST':
        url=request.get_json()
        img_data=url['data'].split(",")
        img_base=base64.b64decode(img_data[1])
        file=open('img.jpg','wb')
        file.write(img_base)
        file.close()
        img=Image.open(r"img.jpg")
        img=img.crop((130,15,300,200))
        img=np.array(img).astype('float32')/255
        img=transform.resize(img,(250,250,3))
        img=np.expand_dims(img,axis=0)
        predictions=model.predict(img)
        class_names=["Human","Not Human"]
        if np.argmax(a=predictions)==0:
            return "ok"
        else:
            return "Please don't move out of the webcam"
    
if __name__=="__main__":
    app.run(debug=True,port=8000)

