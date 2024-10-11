import streamlit as st

import cv2

st.title("Streamlit + OpenCV Sample")

cap = cv2.VideoCapture(0)
st.header("Image")
image_placeholder = st.empty()
if st.button("Start Streaming"):
    while True:
        ret, frame = cap.read()
        if ret:
            image_placeholder.image(frame, channels="BGR")
        
            

cap.release()
#cv2.destroyAllWindows()
