import os

import librosa
import matplotlib.pyplot as plt
import numpy as np

# audio_path = 'E:/Python_Pro/road_rage/dataset/222/20240719165745.wav'
#
# #y, sr = librosa.load(audio_path, sr=16000)
# #time = librosa.times_like(y, sr=sr)
# y, sr = librosa.load(audio_path, sr=16000)
#
# print(sr)
# time = np.arange(0, len(16000*y)) # 计算时间轴，单位为秒
#
# print(time.shape)
# # 绘制振幅图
# plt.figure(figsize=(14, 5))
# plt.plot(time, y, linewidth=0.5)
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Audio Waveform')
# plt.show()

def get_audio(path,sr,start_frame,end_frame):

    y, sr = librosa.load(path, sr=sr,duration=None)
    start_audio=int(start_frame*sr)
    end_audio=int(end_frame*sr)
    return y[start_audio:end_audio]



