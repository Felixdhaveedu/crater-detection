import streamlit as st
from PIL import Image
import io
from ultralytics import YOLO

#st.title(st.config.get_option("pageTitle"))
# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image file as bytes
    image_bytes = uploaded_file.read()
    
    # Display the original image
    st.image(image_bytes, caption="Original Image", use_column_width=True)
    
    # Convert image bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    
    # Resize the image
    new_width = 640
    new_height = 640
    resized_image = image.resize((new_width, new_height))
    model=YOLO("D:\\My Study\\astronomicaldatascience\\internship\\results\\runs_113_epochs\\content\\runs\\detect\\train\\weights\\best.pt")
    result=model(resized_image)  
    
    # Display the resized image
    st.image(result[0].plot(), caption="Resized Image", use_column_width=True)











#import streamlit as st

#st.title('Crater detection')
#st.header('Please upload an image')
#source_img = st.file_uploader('Choose an image...',type=['png','jpeg'])
#print(source_img)