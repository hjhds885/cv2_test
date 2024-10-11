import streamlit as st

import cv2
import os

def set_camera_permissions():
    # カメラデバイスのパスを指定
    camera_device = '/dev/video0'

    # 666権限を設定
    command = f'sudo chmod 666 {camera_device}'

    # コマンドを実行
    os.system(command)

# 関数を呼び出して権限を設定
set_camera_permissions()

st.title("Streamlit + OpenCV Sample")

cap = cv2.VideoCapture(0)
st.header("Image")
image_placeholder = st.empty()

while True:
    ret, frame = cap.read()
    if ret:
        image_placeholder.image(frame, channels="BGR")

cap.release()
#cv2.destroyAllWindows()
