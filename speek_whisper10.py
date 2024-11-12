import streamlit as st
import numpy as np
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from pydub import AudioSegment
import queue, pydub, tempfile,  os, time
import whisper
import librosa
import torch.classes

#AudioSegment.converter = "/usr/bin/ffmpeg"
#pydub.AudioSegment.converter = "/usr/bin/ffmpeg"  #'c:\\FFmpeg\\bin\\ffmpeg.exe'

def save_audio(audio_segment: AudioSegment, base_filename: str) -> None:
    """
    Save an audio segment to a .wav file.
    Args:
        audio_segment (AudioSegment): The audio segment to be saved.
        base_filename (str): The base filename to use for the saved .wav file.
    """
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
    ########################################################
    #無音ファイルを検出し、それをスキップする
    threshold=65 #-50.0
    # 音声データをnumpy配列に変換
    samples = np.array(audio_segment.get_array_of_samples())
    # エネルギーを計算
    energy = np.sqrt(np.mean(samples**2))
    print("energy=",energy)     #55,65で以下の処理をスキップ
    # エネルギーがしきい値以下であるかどうかを確認
    if energy < threshold  :
        print("無音ファイルを検出し、それをスキップする") 
    #######################################################
   

    #########################################################

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        audio_segment.export(tmpfile.name, format="wav")
        file_size = os.path.getsize(tmpfile.name) 
        print("file_size=",file_size)
        if file_size < 390000:
            print("無音ファイルが検出されました。処理をスキップします。") 
            return "無音のため、音声は認識されませんでした。"
        
        #is_silent = is_silent_chunk(audio_segment) 
        #if is_silent:
            #print("サウンドチャンクは無音です:", is_silent)

       ###########################################################     
        # 音声ファイルを読み込む
        audio_path = tmpfile.name  #"audio.wav"
        y, sr = librosa.load(audio_path, sr=None)
        # 無音とみなすエネルギーの閾値（適宜調整）1e-3
        silence_threshold = 1e-4  # 小さい値ほど厳密に無音を検出
        # 短時間エネルギーを計算
        frame_length = 2048  # フレームサイズ
        hop_length = 512  # スライド幅
        energy = np.array([
            sum(abs(y[i:i + frame_length]**2))
            for i in range(0, len(y), hop_length)
        ])
        #print(f"短時間エネルギー: {energy}") 
        # エネルギーが閾値以下のフレームを無音と判断
        is_silent = energy < silence_threshold
        # デバッグ用: 無音フレームの割合を確認
        silent_ratio = np.sum(is_silent) / len(is_silent)
        print(f"無音フレームの割合: {silent_ratio:.2%}") 
        if file_size < 390000:
            print("無音ファイルです。処理をスキップします。") 
            #return "無音のため、音声は認識されませんでした。"
        
        # 無音かどうかの結果を出力
        elif np.all(is_silent):
            print("無音が検出されました。")
            
        else:
            print("音声が含まれています。")
            
            # Whisperのモデルをロード
            model = whisper.load_model("small")  # モデルのサイズは適宜選択
            #base:74M,small:244M,medium,large
            # 一時ファイルのパスを指定
            audio = whisper.load_audio(tmpfile.name)
            audio = whisper.pad_or_trim(audio)

            # 無音を検出するための閾値 0.01 0.05 1.00以下
            silence_threshold1 = 0.30  #0.05
            print("無音レベル＝",np.max(np.abs(audio)))
            # 無音の検出：音声データの振幅が閾値を下回る場合は無音と判断
            #音声信号の最大振幅が非常に低い場合を「無音」と見なしています。
            if np.max(np.abs(audio)) < silence_threshold1:
                print("無音が検出されました。テキストを返さず終了します。")
            else:
                # 無音でない場合のみWhisperでの文字起こしを実行
                # 音声をデコード
                result = model.transcribe(audio, language="ja")  # 日本語を指定
                answer = result['text']
                #print(answer)
                
                # テキスト出力が空、または空白である場合もチェック
                if result["text"].strip() == "":
                    print("テキスト出力が空、または空白である")
                else:
                    answer = result['text']
                    print(answer)
                    return answer
    tmpfile.close()  
    os.remove(tmpfile.name)
    
        ###############################################################
         
def frame_energy(frame):
    """
    Compute the energy of an audio frame.
    Args:
        frame (VideoTransformerBase.Frame): The audio frame to compute the energy of.
    Returns:
        float: The energy of the frame.
    """
    samples = np.frombuffer(frame.to_ndarray().tobytes(), dtype=np.int16)

    # デバッグ用にサンプルの一部を出力 
    #print("Samples:", samples[:10])
    # NaNや無限大の値を除去 
    if not np.isfinite(samples).all(): 
        samples = samples[np.isfinite(samples)]
        #print("Filtered Samples:", samples[:10]) # フィルタリング後のサンプルを出力

    if len(samples) == 0: 
        return 0.0

    energy = np.sqrt(np.mean(samples**2)) 
    #print("Energy:", energy) 
    # エネルギーを出力 
    return energy
 

def process_audio_frames(audio_frames, sound_chunk, silence_frames, energy_threshold):
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
    戻り値：
        tuple[AudioSegment, int]: 更新されたサウンドチャンクと無音フレームの数。

    """
    for audio_frame in audio_frames:
        sound_chunk = add_frame_to_chunk(audio_frame, sound_chunk)

        energy = frame_energy(audio_frame)
        if energy < energy_threshold:
            silence_frames += 1 #無音のエネルギーしきい値以下である場合、無音フレームの数を1つ増やします。
        else:
            silence_frames = 0 #エネルギーがしきい値を超える場合、無音フレームをリセットして0にします。

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
        status_indicator,
        text_output,
        timeout=3, 
        energy_threshold=2000,  #2000
        silence_frames_threshold=100,   #500  #100  #1024でGood！！
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
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": False, "audio": True},
    )

    sound_chunk = pydub.AudioSegment.empty()
    silence_frames = 0

    while True:
        if webrtc_ctx.audio_receiver:
            status_indicator.write("🤖何か話して!")

            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=timeout)
            except queue.Empty:
                status_indicator.write("No frame arrived.")
                sound_chunk = handle_queue_empty(sound_chunk, text_output)
                continue
            sound_chunk, silence_frames = process_audio_frames(audio_frames, sound_chunk, silence_frames, energy_threshold)
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
    #この修正では、asyncio.run(main())を使用してmain()関数を実行することで、
    # イベントループを正しく管理しています。これにより、NoneTypeオブジェクトに対するエラーが発生する可能性が減少します。
    #asyncio.run(main())
