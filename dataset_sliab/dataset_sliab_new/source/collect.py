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
    def __init__(self, frame_rate,dir):
        os.makedirs(os.path.join(dir,"vision"), exist_ok=True)
        self.frame_rate = frame_rate
        self.cap = cv2.VideoCapture(1)
        self.save_dir = os.path.join(dir,"vision")
        self.running = True
        self.count = 0
        self.save_event = threading.Event()  # 创建一个事件用于控制保存
        self.timestamp=''
    def run(self):
        while self.running:
            self.save_event.wait()  # 等待事件被设置
            self.save_data()  # 保存数据
            if self.count >= 2:  # 达到2张后，重置计数并清除事件
                self.count = 0
                self.save_event.clear()  # 清除事件，暂停保存
    def save_data(self):
        ret, frame = self.cap.read()
        if ret:
            file_time = self.timestamp
            file_name = f"{file_time}_{self.count}.jpg"
            file_path = os.path.join(self.save_dir, file_name)  # 保存路径
            cv2.imwrite(file_path, frame)  # 保存图像
            # print("Successfully saved:", file_name)
            self.count += 1
    def stop(self):
        self.running = False
        self.cap.release()  # 释放摄像头资源
        cv2.destroyAllWindows()  # 关闭所有OpenCV窗口

# class CollectSilab:
#     def __init__(self, ip,port,frame_rate,dir):
#         os.makedirs(os.path.join(dir, "silab"), exist_ok=True)
#         self.frame_rate = frame_rate
#         self.timestamp = ''
#         self.sensor_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         self.sensor_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024)
#         self.sensor_socket.bind((ip, port))
#         self.sensor_socket.listen(1)
#         self.conn, addr = self.sensor_socket.accept()
#         self.running = True
#         self.count = 0
#         self.save_event = threading.Event()  # 创建一个事件用于控制保存
#         self.save_dir = os.path.join(dir,"silab")
#         self.csv_file_path = os.path.join(self.save_dir, 'silab_data04.csv')
#         with open(self.csv_file_path, mode='w', newline='') as csv_file:
#             writer = csv.writer(csv_file)
#             writer.writerow(['Timestamp', 'Data'])  # 写入列标题
#     def run(self):
#         while self.running:
#             self.save_event.wait()
#             self.savedata()
#             # if self.count >= 10:  # 达到10帧数后，重置计数并清除事件
#             #     self.count = 0
#             #     self.save_event.clear()  # 清除事件，暂停保存
#             self.save_event.clear()
#     def savedata(self):
#
#         data = self.conn.recv(99999).decode("utf-8")  # tcp接收
#         print(data)
#         with open(self.csv_file_path, mode='a', newline='') as csv_file:
#             writer = csv.writer(csv_file)
#             writer.writerow([self.timestamp, data])  # 写入时间戳和数据
#         # self.count+=1
#
#
#     def stop(self):
#         self.running = False
#         self.sensor_socket.close()  # 关闭套接字

class AudioCollector:
    def __init__(self, dir):
        os.makedirs(os.path.join(dir, "audio"), exist_ok=True)
        self.output_file = None
        self.timestamp=''
        self.save_event = threading.Event()  # 创建一个事件用于控制保存
        self.dir=os.path.join(dir,"audio")
        self.chunk = 1024
        self.sample_format = pyaudio.paInt16
        self.channels = 2
        self.fs = 16000
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.sample_format, channels=self.channels,
                                   rate=self.fs, frames_per_buffer=self.chunk,input=True)
        self.frames = []

    def run(self):
        self.save_event.wait()
        self.output_file = os.path.join(self.dir, f"{self.timestamp}.wav")
        while True:
            data = self.stream.read(self.chunk)
            self.frames.append(data)
            # 可以在这里根据需要处理保存逻辑

    def stop(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        # 保存录音
        with wave.open(self.output_file, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(self.sample_format))
            wf.setframerate(self.fs)
            wf.writeframes(b''.join(self.frames))


class CollectSilab:
    def __init__(self, ip, port, frame_rate, dir):
        os.makedirs(os.path.join(dir, "silab"), exist_ok=True)
        self.frame_rate = frame_rate
        self.timestamp = ''
        self.sensor_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sensor_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024)
        self.sensor_socket.bind((ip, port))
        self.sensor_socket.listen(1)
        self.conn, addr = self.sensor_socket.accept()
        self.running = True
        self.data_buffer = []  # 用于存储接收到的数据
        self.save_event = threading.Event()  # 创建一个事件用于控制保存
        self.save_dir = os.path.join(dir, "silab")
        self.csv_file_path = os.path.join(self.save_dir, 'silab_data04.csv')

        with open(self.csv_file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Timestamp', 'Data'])  # 写入列标题

        # 启动数据接收线程
        self.data_thread = threading.Thread(target=self.receive_data, daemon=True)
        self.data_thread.start()

    def run(self):
        last_save_time = time.time()  # 记录上一次保存的时间
        while self.running:
            current_time = time.time()
            if current_time - last_save_time >= 1:  # 每秒保存一次
                self.save_data()  # 保存数据
                last_save_time = current_time  # 更新上一次保存的时间

    def receive_data(self):
        while self.running:
            try:
                data = self.conn.recv(99999)
                if not data:
                    break  # 如果没有数据，则跳出循环
                self.data_buffer.append(data.decode("utf-8"))  # 将接收到的数据添加到缓冲区
            except Exception as e:
                print(f"Error receiving data: {e}")
                break

    def save_data(self):
        if self.data_buffer:  # 如果缓冲区有数据
            timestamp = time.strftime('%Y%m%d%H%M%S')  # 获取当前时间戳
            data_to_save = "\n".join(self.data_buffer)  # 将缓冲区中的数据合并成字符串
            print(f"{timestamp}: {data_to_save}")
            with open(self.csv_file_path, mode='a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([timestamp, data_to_save])  # 写入时间戳和合并后的数据
            self.data_buffer.clear()  # 清空缓冲区

    def stop(self):
        self.running = False
        self.conn.close()  # 关闭连接
        self.sensor_socket.close()  # 关闭套接字



if __name__ == '__main__':

    directory_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),"data","huanghao_scene_01")

    # 创建目录（如果不存在的话）
    os.makedirs(directory_path, exist_ok=True)


    image_fps = 2  # 图像帧率
    collectImage = CollectImage(image_fps,dir=directory_path)
    collectSilab=CollectSilab("192.168.2.36",8056,10,dir=directory_path)
    #collectSilab = CollectSilab("127.0.0.1", 8050, 10, dir=directory_path)
    collectAudio= AudioCollector(dir=directory_path)
    # 开启线程
    image_thread = threading.Thread(target=collectImage.run)
    image_thread.start()
    silab_thread=threading.Thread(target=collectSilab.run)
    silab_thread.start()

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
                collectAudio.timestamp=collectSilab.timestamp=collectImage.timestamp=time.strftime('%Y%m%d%H%M%S')
                collectImage.save_event.set()  # 设置事件，允许保存图像
                collectSilab.save_event.set()
                collectAudio.save_event.set()
                last_printed_second = current_time  # 更新已打印的秒数
                if keyboard.is_pressed('esc'):
                    break  # 按下 "q" 键，退出循环

    except KeyboardInterrupt:
        pass  # 捕获中断，进行正常停止

    finally:
        collectImage.stop()  # 停止图像采集
        collectAudio.stop()  # 停止音频采集
        collectSilab.stop()
        image_thread.join()  # 等待线程结束
        audio_thread.join()
        silab_thread.join()
        print("数据采集已停止")
