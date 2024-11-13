import streamlit as st
import numpy as np
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from pydub import AudioSegment
import queue, pydub, tempfile,  os, time
import whisper
import torch
import torchaudio
import torchvision

def save_audio(audio_segment: AudioSegment, base_filename: str) -> None:
    filename = f"{base_filename}_{int(time.time())}.wav"
    audio_segment.export(filename, format="wav")

def transcribe(audio_segment: AudioSegment, debug: bool = False) -> str:
    """
    Transcribe an audio segment using OpenAI's Whisper ASR system.
    Args:
        audio_segment (AudioSegment): The audio segment to transcribe.
        debug (bool): If True, save the audio segment for debugging purposes.
    Returns:
        str: The transcribed text.
    """
    if debug:
        save_audio(audio_segment, "debug_audio")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        audio_segment.export(tmpfile.name, format="wav")
        # 一時ファイルのパスを指定
        audio = whisper.load_audio(tmpfile.name)
        audio = whisper.pad_or_trim(audio)
        # Whisperのモデルをロード
        model = whisper.load_model("small")  # モデルのサイズは適宜選択
        #base:74M,small:244M,medium,large
        # 音声をデコード
        result = model.transcribe(audio, language="ja")  # 日本語を指定
        answer = result['text']
        # テキスト出力が空、または空白である場合もチェック
        if answer == "" :
            print("テキスト出力が空")
            return None 
        elif "ご視聴" in answer or "お疲れ様" in answer:
            print("テキスト出力が「ご視聴」、または「お疲れ様」を含む")
            return None 
        else:
            print(answer)
            return answer
    tmpfile.close()  
    os.remove(tmpfile.name)
    
        ###############################################################
         
def frame_energy(frame):
    samples = np.frombuffer(frame.to_ndarray().tobytes(), dtype=np.int16)
    # デバッグ用にサンプルの一部を出力 
    #print("Samples:", samples[:10])
    # NaNや無限大の値を除去 
    #if not np.isfinite(samples).all(): 
        #samples = samples[np.isfinite(samples)]
    #np.isfinite() で無効な値をフィルタリングするだけでは、
    # 空配列のエラーが再び発生する可能性があるため、
    # np.nan_to_num を使用したほうが安全に処理できます。
    # 無効な値を安全な値に置換
    samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    if len(samples) == 0: 
        return 0.0
    energy = np.sqrt(np.mean(samples**2)) 
    #print("Energy:", energy) 
    # エネルギーを出力 
    return energy

def is_silent_frame(audio_frame, amp_threshold):
    """
    フレームが無音かどうかを最大振幅で判定する関数。
    """
    samples = np.frombuffer(audio_frame.to_ndarray().tobytes(), dtype=np.int16)
    max_amplitude = np.max(np.abs(samples))
    return max_amplitude < amp_threshold

def process_audio_frames(audio_frames, sound_chunk, silence_frames, energy_threshold, amp_threshold):
    """
    音声フレームを順次処理し、無音フレームの数をカウントすることです。
    無音フレームが一定数以上続いた場合、無音区間として処理し、後続の処理（例えば、音声認識のトリガー）に役立てます。
    この処理により、無音や音声の有無を正確に検出することができます。

    音声フレームのリストを処理します。 
    引数：
        audio_frames (list[VideoTransformerBase.Frame]): 処理する音声フレームのリスト。
        sound_chunk (AudioSegment): 現在のサウンドチャンク。
        silence_frames (int): 現在の無音フレームの数。
        energy_threshold (int): 無音検出に使用するエネルギーしきい値。
        amp_threshold:無音検出に使用する最大振幅しきい値。
    戻り値：
        tuple[AudioSegment, int]: 更新されたサウンドチャンクと無音フレームの数。

    """
    for audio_frame in audio_frames:
        sound_chunk = add_frame_to_chunk(audio_frame, sound_chunk)

        energy = frame_energy(audio_frame)
        
        if energy < energy_threshold or is_silent_frame(audio_frame, amp_threshold):
            silence_frames += 1 
            #無音のエネルギー又は最大振幅がしきい値以下である場合、無音フレームの数を1つ増やします。
        else:
            silence_frames = 0 
            #エネルギー又は最大振幅がしきい値を超える場合、無音フレームをリセットして0にします。

    return sound_chunk, silence_frames

def add_frame_to_chunk(audio_frame, sound_chunk):
    """
    オーディオフレームをサウンドチャンクに追加します。 
    引数：
        audio_frame (VideoTransformerBase.Frame): 追加するオーディオフレーム。
        sound_chunk (AudioSegment): 現在のサウンドチャンク。
    戻り値：
        AudioSegment: 更新されたサウンドチャンク。
   
    """
    sound = pydub.AudioSegment(
        data=audio_frame.to_ndarray().tobytes(),
        sample_width=audio_frame.format.bytes,
        frame_rate=audio_frame.sample_rate,
        channels=len(audio_frame.layout.channels),
    )
    sound_chunk += sound
    return sound_chunk

def handle_silence(sound_chunk, silence_frames, silence_frames_threshold, text_output):
    """
    オーディオストリーム内の無音を処理します。 
    引数：
        sound_chunk (AudioSegment): 現在のサウンドチャンク。
        silence_frames (int): 現在の無音フレームの数。
        silence_frames_threshold (int): 無音フレームのしきい値。
        text_output (st.empty): Streamlitのテキスト出力オブジェクト。
    戻り値：
        tuple[AudioSegment, int]: 更新されたサウンドチャンクと無音フレームの数。
   
    """
    if silence_frames >= silence_frames_threshold: 
        #無音フレーム数が100以上の時、音声の途切れ（間隔）として扱う
        if len(sound_chunk) > 0:
            #if is_silent_chunk(sound_chunk):
                #print("サウンドチャンク内のすべてのフレームが無音フレームです")
            #else: 
            #print("handle_silenceルーチン")
            frame_length_ms = 20 
            num_frames = len(sound_chunk) // frame_length_ms 
            #print("全フレーム数＝",num_frames)
            #print("無音フレーム数＝",silence_frames)

            if num_frames-silence_frames > 100:
                text = transcribe(sound_chunk)
                text_output.write(text)
            #print("オーディオストリーム内の無音時の応答=",text)
            sound_chunk = pydub.AudioSegment.empty()
            silence_frames = 0

       #else:
            #print("無音フレーム数が少ないhandle_silenceルーチン")
            #text = transcribe(sound_chunk)
            #text_output.write(text)

    return sound_chunk, silence_frames

def handle_queue_empty(sound_chunk, text_output):
    """
    Handle the case where the audio frame queue is empty.
    Args:
        sound_chunk (AudioSegment): The current sound chunk.
        text_output (st.empty): The Streamlit text output object.
    Returns:
        AudioSegment: The updated sound chunk.
    """
    if len(sound_chunk) > 0:
        print("handle_queue_emptyルーチン")   
        text = transcribe(sound_chunk)
        text_output.write(text)
        sound_chunk = pydub.AudioSegment.empty()

    return sound_chunk

def app_sst(
        #audio_receiver_size,
        status_indicator,
        text_output,
        #energy_threshold, #=2000,  #2000
        #amp_threshold,
        #silence_frames_threshold,   #=100 #500  #100  #1024でGood！！
        #timeout=1, #3, 
        ):
    """
    リアルタイム音声認識のためのメインアプリケーション関数。 
    この関数はWebRTCストリーマーを作成し、音声データの受信を開始し、
    音声フレームを処理し、一定のしきい値を超える無音が続いた場合に音声をテキストに変換します。
    引数：
        status_indicator: ステータス（実行中または停止中）を表示するためのStreamlitオブジェクト。
        text_output: 認識されたテキストを表示するためのStreamlitオブジェクト。
        timeout (int, オプション): オーディオレシーバーからフレームを取得するためのタイムアウト。デフォルトは3秒。
        energy_threshold (int, オプション): フレームが無音と見なされるエネルギーしきい値。デフォルトは2000。
        silence_frames_threshold (int, オプション): トランスクリプションをトリガーするための連続無音フレームの数。デフォルトは100フレーム。
    """

    audio_receiver_size = st.sidebar.slider(
        "audio_receiver_size(処理音声フレーム数。デフォルト512):", 
        min_value=64, max_value=1024, value=512, step=64
    )
    energy_threshold = st.sidebar.slider(
        "energy_threshold(無音エネルギーしきい値。デフォルト2000):", 
        min_value=100, max_value=5000, value=2000, step=100
    )
    amp_threshold = st.sidebar.slider(
        "amp_threshold(無音最大振幅しきい値。デフォルト0.3):", 
        min_value=0.00, max_value=1.00, value=0.30, step=0.05
    )
    # 無音を検出するための閾値 0.01 0.05 1.00以下
        #amp_threshold = 0.30  #0.05
    silence_frames_threshold = st.sidebar.slider(
        "silence_frames_threshold(トリガー用連続無音フレーム数。デフォルト100):", 
        min_value=20, max_value=300, value=60, step=20
    )
    #60がBest,デフォルト100
    timeout = st.sidebar.slider(
        "timeout(フレームを取得するためのタイムアウト。デフォルト3秒):", 
        min_value=1, max_value=3, value=1, step=1
    )
    #stで使う変数初期設定
    #st.session_state.audio_receiver_size = audio_receiver_size
    st.session_state.energy_threshold = energy_threshold
    st.session_state.amp_threshold = amp_threshold
    st.session_state.silence_frames_threshold = silence_frames_threshold
    st.session_state.timeout = timeout
    
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        desired_playing_state=True, 
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=audio_receiver_size, 
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False, "audio": True},
    )

    sound_chunk = pydub.AudioSegment.empty()
    silence_frames = 0

    while True:
        if webrtc_ctx.audio_receiver:
            status_indicator.write("🤖何か話して!")

            timeout=st.session_state.timeout
            energy_threshold=st.session_state.energy_threshold
            amp_threshold=st.session_state.amp_threshold
            silence_frames_threshold= st.session_state.silence_frames_threshold
            #print("audio_receiver_size =",audio_receiver_size)
            
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=timeout)
                #print("timeout=",timeout)
            except queue.Empty:
                status_indicator.write("No frame arrived.")
                sound_chunk = handle_queue_empty(sound_chunk, text_output)
                continue
            sound_chunk, silence_frames = process_audio_frames(audio_frames, sound_chunk, silence_frames, energy_threshold,amp_threshold)
            sound_chunk, silence_frames = handle_silence(sound_chunk, silence_frames, silence_frames_threshold, text_output)
        else:
            status_indicator.write("Stopping.")
            if len(sound_chunk) > 0:
                #print("len(sound_chunk)=",len(sound_chunk))
                text = transcribe(sound_chunk.raw_data)  #?
                text_output.write(text)
            break

def main():
    st.title("Real-time Speech-to-Text")
    
    
    status_indicator = st.empty()
    text_output = st.empty()
    app_sst(status_indicator,text_output)

if __name__ == "__main__":
    main()
