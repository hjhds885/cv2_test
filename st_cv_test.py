import streamlit as st

import cv2

st.title("Streamlit + OpenCV Sample")

cap = cv2.VideoCapture(0)
st.header("Image")
image_placeholder = st.empty()
while True:
    ret, frame = cap.read()
    if not ret:
        st.warning("Could not read frame. Check camera connectivity.")
        break
    image_placeholder.image(frame, channels="BGR")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
#cv2.destroyAllWindows()
