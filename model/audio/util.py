import numpy as np
import librosa
from matplotlib import pyplot as plt

#平均mfcc值
def mean_mfcc_eval(data, sr, n_mfcc, n_fft=2048, hop_length=512):
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean
#时间、特征的mfcc
def mfcc_eval(data, sr, n_mfcc, n_fft=2048, hop_length=512):
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfccs=mfccs.T
    return mfccs


def mfcc_piptrack(data):
    stft = np.abs(librosa.stft(data))
    print(stft.shape)
    # fmin 和 fmax 对应于人类语音的最小最大基本频率
    pitches, magnitudes = librosa.piptrack(y=data, sr=16000, S=stft, fmin=70, fmax=400)
    pitch = []
    for i in range(magnitudes.shape[1]):
        index = magnitudes[:, 1].argmax()
        pitch.append(pitches[index, i])

    pitch_tuning_offset = librosa.pitch_tuning(pitches)

    return pitch_tuning_offset



def get_audio(path,sr,start_frame,end_frame):

    y, sr = librosa.load(path, sr=sr,duration=None)
    start_audio=int(start_frame*sr)
    end_audio=int(end_frame*sr)
    return y[start_audio:end_audio]

if __name__ == '__main__':
    audio_data = np.random.rand(16000 * 20)  # 示例：10秒的随机数据

    mfcc_data=mfcc_eval(audio_data, sr=16000, n_mfcc=128)
    # print(mfcc_data.shape)

    path, sr = 'E:/Python_Pro/road_rage/dataset/audio/20240719165745.wav', 16000

    data=get_audio(path,sr=16000,start_frame=0,end_frame=2)
    piptrack_data=mfcc_piptrack(audio_data)
    print(piptrack_data)

    # def generate_sine_wave(freq, duration, sr, amplitude=0.5):
    #     t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    #     return amplitude * np.sin(2 * np.pi * freq * t)
    #
    #
    # def generate_cos_wave(freq, duration, sr, amplitude=0.5):
    #     t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    #     return amplitude * np.cos(2 * np.pi * freq * t)
    # 示例使用
    # 假设 'audio_data' 是已加载的音频数据，'sample_rate' 是采样率
    # audio_data = np.random.rand(16000 * 20)  # 示例：10秒的随机数据
    # sample_rate = 16000  # 示例采样率
    # frequency = 440  # Hz
    # duration = 2  # 秒
    #
    # audio_data_with_sound = generate_sine_wave(frequency, duration, sample_rate)
    #
    # # 生成无声音的音频数据（全零数组）
    # audio_data_low_volume = generate_cos_wave(900, duration, sample_rate, amplitude=0.5)
    # 打印特征的形状
    # 可视化 MFCC 特征
    # plt.figure(figsize=(10, 6))
    # plt.bar(range(len(mfcc_features)), mfcc_features)
    # plt.title('MFCC')
    # plt.show()
    # plt.figure(figsize=(10, 6))
    # plt.bar(range(len(mfcc_features2)), mfcc_features2)
    # plt.title('MFCC2')
    # plt.show()