import streamlit as st
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def recv(self, frame):   
        try:
            self.frame = frame.to_ndarray(format="bgr24")
            img = frame.to_ndarray(format="bgr24")

            #img = cv2.cvtColor(cv2.Canny(img, self.threshold1, self.threshold2), cv2.COLOR_GRAY2BGR)
            frame = av.VideoFrame.from_ndarray(img, format="bgr24")
        except AttributeError as e:
            st.error(f"Error converting frame: {e}")
        return frame

st.title("Webカメラの認識デモ")

# WebRTCを使用してカメラにアクセス
st.header("Webcam Stream")
webrtc_ctx=webrtc_streamer(
    key="example",
    #desired_playing_state=True, 
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    #video_transformer_factory=VideoTransformer 廃止
    video_processor_factory=VideoTransformer,
)
if webrtc_ctx.state.playing:
    st.write("WebRTC is playing")
else:
    st.write("WebRTC is not playing")
