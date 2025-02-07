import streamlit as st
from tensorflow.keras.models import Sequential, model_from_json
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import UnidentifiedImageError
from PIL import Image

json_file = open('flower.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights(r"D:\Machine Learning\Deep Learning\flowers_folder\flower.h5")

st.title('Flower Classification Using CNN')

class_name = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']

genre = st.radio(
    "How you want to  upload your image",
    ('Browse Photos', 'Camera'))

if genre == 'Browse Photos':
    ImagePath = st.camera_input("Take a picture")

else:
    ImagePath = st.file_uploader(" Choose a file ")

if ImagePath is not None:
    try:
        image_ = Image.open(ImagePath)

        st.image(image_, width=250)
        
    except UnidentifiedImageError:
        st.write('Inout Valid file format!! [jpeg, jpg, png only this file format is supported]')


try:
    if st.button('Predict'):
        test_image = image.load_img(ImagePath, target_size= (256, 256))
        test_image = image.img_to_array(test_image)

        test_image = np.expand_dims(test_image, axis=0)

        result = loaded_model.predict(test_image, verbose=0)

        type_ = class_name[np.argmax(result)]
        st.header('Prediction is: '+ type_)
        st.header('Confidence is: '+ str(round(np.max(result),4)*100)+ '%')

except TypeError:
    st.header('Please Upload Your File!!')

except UnidentifiedImageError:
    st.header('Input Valid File!!')