import streamlit as st

import cv2

st.title("Streamlit + OpenCV Sample")

cap = cv2.VideoCapture(0)
st.header("Image")
image_placeholder = st.empty()
while True:
    ret, frame = cap.read()
    if ret:
        image_placeholder.image(frame, channels="BGR")
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    # Streamlitのst.stop()を使ってストリームを終了できるようにする
    if st.button("Stop Streaming"):
        break
        

cap.release()
#cv2.destroyAllWindows()
