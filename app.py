import matplotlib.pyplot as plt
import numpy as np
import easyocr
import pandas as pd
import cv2
import os
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import streamlit as st

# Loading model
model = load_model('my_model.h5')

# Object detection
def object_detection(path):
    image = load_img(path)
    image = np.array(image, dtype=np.uint8)
    image1 = load_img(path, target_size=(224, 224))
    img_arr_224 = img_to_array(image1) / 255.0
    h, w, d = image.shape
    test_arr = img_arr_224.reshape(1, 224, 224, 3)
    coords = model.predict(test_arr)
    denom = np.array([w, w, h, h])
    coords = coords * denom
    coords = coords.astype(np.int32)
    xmin, xmax, ymin, ymax = coords[0]
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)
    cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)
    plt.figure(figsize=(10, 8))
    plt.imshow(image)
    plt.show()
    return image,coords

# OCR
def ocr(path):
    img = np.array(load_img(path))
    image,coords = object_detection(path)
    xmin, xmax, ymin, ymax = coords[0]
    roi = img[ymin:ymax, xmin:xmax]
    reader = easyocr.Reader(['en'])
    result = reader.readtext(roi)
    plt.imshow(roi)
    plt.axis('off')
    plt.show()
    try:
        result = reader.readtext(roi)
        return roi, result[0][1]
    except IndexError:
        return roi, "No text detected"
    #return roi,result[0][1]
#saving the output    
def save_to_csv(image_path, number_plate):
    csv_data = {'Image': [image_path], 'Number_Plate': [number_plate]}
    df = pd.DataFrame(csv_data)
    if not os.path.exists('extracted_data.csv'):
        df.to_csv('extracted_data.csv', index=False)
    else:
        df.to_csv('extracted_data.csv', mode='a', header=False, index=False)    
# main application
def main():
    st.title("license Number Plate Extration App🚖")
    st.write("Upload an image and the app will perform extraction on it.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        #image = Image.open(uploaded_file)
        original_image = Image.open(uploaded_file)
        st.image(original_image, caption='Image without detection', use_column_width=True)
        if st.button("Perform OCR"):
            image,coords=object_detection(uploaded_file)
            st.image(image, caption='Image with detection', use_column_width=True)
            img,result=ocr(uploaded_file)
            st.image(img, caption='Image of License Plate', use_column_width=True)
            st.write("OCR Result:", result)
            save_to_csv(img,result)
            
                 
            

if __name__ == "__main__":
    main()
