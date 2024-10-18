import csv
import os
import socket

import cv2
import threading
import time

import keyboard
import pyaudio
import wave

class CollectImage:
    def __init__(self, frame_rate):

        self.frame_rate = frame_rate
        self.cap = cv2.VideoCapture(0)

        self.running = True
        self.count = 0
        self.save_event = threading.Event()  # 创建一个事件用于控制保存
        self.timestamp=''
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.output_file = 'output.avi'
        self.frames = []
    def run(self):
        while self.running:
            self.save_event.wait()  # 等待事件被设置
            self.save_data()  # 保存数据
            if self.count >= 10:  # 达到2张后，重置计数并清除事件
                self.count = 0
                self.save_event.clear()  # 清除事件，暂停保存
    def save_data(self):
        ret, frame = self.cap.read()
        if ret:
            self.frames.append(frame)
            # print("Successfully saved:", file_name)
            self.count += 1
    def stop(self):
        self.running = False
        self.cap.release()  # 释放摄像头资源
        self.write_video()  # 一次性写入视频文件
        cv2.destroyAllWindows()  # 关闭所有OpenCV窗口

    def write_video(self):
        out = cv2.VideoWriter(self.output_file, self.fourcc, self.frame_rate, (640, 480))
        for frame in self.frames:
            out.write(frame)  # 一次性写入所有帧
        out.release()  # 释放视频写入对象

class AudioCollector:
    def __init__(self):
        self.output_file = None
        self.timestamp = ''
        self.save_event = threading.Event()
        self.chunk = 1024
        self.sample_format = pyaudio.paInt16
        self.channels = 2
        self.fs = 16000
        self.p = pyaudio.PyAudio()

        # 尝试打开音频流
        try:
            self.stream = self.p.open(format=self.sample_format, channels=self.channels,
                                       rate=self.fs, frames_per_buffer=self.chunk, input=True)
            print("Audio stream opened successfully.")
        except Exception as e:
            print(f"Error opening audio stream: {e}")
            self.stream = None
            return  # 如果流无法打开，返回

        self.frames = []

    def run(self):
        while self.stream:  # 只在流有效时运行
            self.save_event.wait()
            if not self.output_file:
                self.output_file = f"{self.timestamp}.wav"
            try:
                data = self.stream.read(self.chunk)
                self.frames.append(data)
            except IOError as e:
                print(f"Error reading audio stream: {e}")
                break  # 如果读取失败，退出循环

    def stop(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.p.terminate()
            with wave.open(self.output_file, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.p.get_sample_size(self.sample_format))
                wf.setframerate(self.fs)
                wf.writeframes(b''.join(self.frames))
            print("Audio collection stopped and saved.")




if __name__ == '__main__':
    image_fps = 10 # 图像帧率
    collectImage = CollectImage(image_fps)

    collectAudio= AudioCollector()
    # 开启线程
    image_thread = threading.Thread(target=collectImage.run)
    image_thread.start()


    audio_thread = threading.Thread(target=collectAudio.run)
    audio_thread.start()
    #控制时间
    last_printed_second = int(time.time())
    try:
        while True:
            current_time = int(time.time())  # 获取 当前的整秒时间
            if current_time > last_printed_second:  # 检查是否到达下一秒
                timestamp= time.strftime('%Y%m%d%H%M%S')
                print(timestamp)
                collectAudio.timestamp=collectImage.timestamp=time.strftime('%Y%m%d%H%M%S')
                collectImage.save_event.set()  # 设置事件，允许保存图像

                collectAudio.save_event.set()
                last_printed_second = current_time  # 更新已打印的秒数
                if keyboard.is_pressed('esc'):
                    break  # 按下 "q" 键，退出循环

    except KeyboardInterrupt:
        pass  # 捕获中断，进行正常停止

    finally:
        collectImage.stop()  # 停止图像采集
        collectAudio.stop()  # 停止音频采集
        image_thread.join()  # 等待线程结束
        audio_thread.join()
        print("数据采集已停止")

#ffmpeg -i output.avi -i output.wav -c:v copy -c:a aac output_final.mp4
