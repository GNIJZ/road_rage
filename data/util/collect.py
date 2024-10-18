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
    def __init__(self, frame_rate,dir,save):
        os.makedirs(os.path.join(dir,"vision"), exist_ok=True)
        self.frame_rate = frame_rate
        self.cap = cv2.VideoCapture(0)
        self.save_dir = os.path.join(dir,"vision")
        self.running = True
        self.count = 0
        self.save_event = threading.Event()  # 创建一个事件用于控制保存
        self.timestamp=''
        self.save = save
    def run(self):
        while self.running:
            self.save_event.wait()  # 等待事件被设置
            if self.save:
                self.save_data()  # 保存数据
            else:
                print("kaifazhong")
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



class AudioCollector:
    def __init__(self, dir,save):
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
        self.save=save
    def run(self):
        self.save_event.wait()  # 等待保存事件触发
        self.output_file = os.path.join(self.dir, f"{self.timestamp}.wav")

        while self.save_event.is_set():  # 在事件被设置的情况下循环
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)  # 捕获溢出异常
                self.frames.append(data)
            except IOError as e:  # 捕获输入输出错误
                print(f"Error reading stream: {e}")
                break  # 如果发生错误，退出循环

    def stop(self):
        self.save_event.clear()  # 确保事件被清除，停止读取
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        # 保存录音
        if self.save:
            with wave.open(self.output_file, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.p.get_sample_size(self.sample_format))
                wf.setframerate(self.fs)
                wf.writeframes(b''.join(self.frames))
        else:
            print("语音不保存")
# 单独线程的tcp
"""
class CollectSilab:
    def __init__(self, ip,port,frame_rate,dir):
        os.makedirs(os.path.join(dir, "silab"), exist_ok=True)
        self.frame_rate = frame_rate
        self.timestamp = ''
        self.sensor_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sensor_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024)
        self.sensor_socket.bind((ip, port))
        self.sensor_socket.listen(1)
        self.conn, addr = self.sensor_socket.accept()
        self.running = True
        self.count = 0
        self.save_event = threading.Event()  # 创建一个事件用于控制保存
        self.save_dir = os.path.join(dir,"silab")
        self.csv_file_path = os.path.join(self.save_dir, 'silab_data04.csv')
        with open(self.csv_file_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Timestamp', 'Data'])  # 写入列标题
    def run(self):
        while self.running:
            self.save_event.wait()
            self.savedata()
            # if self.count >= 10:  # 达到10帧数后，重置计数并清除事件
            #     self.count = 0
            #     self.save_event.clear()  # 清除事件，暂停保存
            self.save_event.clear()
    def savedata(self):

        data = self.conn.recv(99999).decode("utf-8")  # tcp接收
        print(data)
        with open(self.csv_file_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([self.timestamp, data])  # 写入时间戳和数据
        # self.count+=1
    def stop(self):
        self.running = False
        self.sensor_socket.close()  # 关闭套接字
"""

#原始tcp接收代码
class CollectSilab:
    # 平均取数
    """
    savedata方法中的可能需要的平均取10帧：
        groups = [data[i:i + 15] for i in range(0, len(data), 15)]
        # 如果总组数少于10个，直接返回所有组
        if len(groups) <= 10:
            return groups
        # 计算每个选择的数据组的间隔
        interval = len(groups) // 10  # 每隔多少个取一个
        # 使用间隔选择10个数据组
        selected_data = [groups[i] for i in range(0, len(groups), interval)][:10]
        # 打印结果
        for index, group in enumerate(selected_data):
            print(f'Group {index + 1}: {group}')
    """
    def __init__(self, ip,port,frame_rate,dir,save):
        os.makedirs(os.path.join(dir, "silab"), exist_ok=True)
        self.frame_rate = frame_rate
        self.timestamp = ''
        self.sensor_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sensor_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024*1024)
        self.sensor_socket.bind((ip, port))
        self.sensor_socket.listen(1)
        self.conn, addr = self.sensor_socket.accept()
        self.running = True
        self.save_event = threading.Event()  # 创建一个事件用于控制保存
        self.save = save
        self.save_dir = os.path.join(dir,"silab")
        self.csv_file_path = os.path.join(self.save_dir, 'silab_data.csv')
        #缓冲区
        self.buffered_data = []  # 使用缓冲区收集数据
        if save:
            self.writer = open(self.csv_file_path, mode='w', newline='')  # 直接打开文件
            self.csv_writer = csv.writer(self.writer)  # 使用文件对象创建csv.writer
            self.csv_writer.writerow(['Timestamp', '刹车', '油门', '方向盘'])  # 写入列标题

    def run(self):
        while self.running:
            self.save_event.wait()
            self.savedata()
            self.save_event.clear()
    def savedata(self):
        data = self.conn.recv(5000).decode("utf-8")  # tcp接收
        #print(data)
        data=data[-150:]
        processed_data = []
        for i in range(0, len(data), 15):  # 每15个字符一组
            group = data[i:i + 15]
            if len(group) == 15:
                cols = [group[j:j + 5] for j in range(0, 15, 5)]  # 使用列表推导简化
                # 将时间戳和列数据合并成一行
                processed_data.append([self.timestamp] + cols)
        if self.save:
            self.buffered_data.extend(processed_data)
            if len(self.buffered_data) >= 10:  # 假设每10行写入一次
                self.csv_writer.writerows(self.buffered_data)  # 批量写入
                self.buffered_data.clear()  # 清空缓冲区
    def stop(self):
        self.running = False
        if self.save:
            self.csv_writer.writerows(self.buffered_data)  # 写入剩余数据
            self.writer.close()  # 关闭文件
        self.conn.close()  # 关闭连接
        self.sensor_socket.close()  # 关闭套接字


if __name__ == '__main__':

    directory_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)),"dataset_silab","data","gnij02")
    # 创建目录（如果不存在的话）
    os.makedirs(directory_path, exist_ok=True)

    image_fps = 2  # 图像帧率
    collectImage = CollectImage(image_fps,dir=directory_path,save=False)
    collectSilab=CollectSilab("127.0.0.1",8051,10,dir=directory_path,save=True)
    collectAudio= AudioCollector(dir=directory_path,save=False)

    # 开启线程
    image_thread = threading.Thread(target=collectImage.run)
    image_thread.start()
    silab_thread=threading.Thread(target=collectSilab.run)
    silab_thread.start()
    audio_thread = threading.Thread(target=collectAudio.run)
    audio_thread.start()
    #控制时间
    last_printed_second = int(time.time())
    print(last_printed_second)
    try:
        while True:
            current_time = int(time.time())  # 获取 当前的整秒时间
            if current_time > last_printed_second:  # 检查是否到达下一秒
                timestamp= time.strftime('%Y%m%d%H%M%S')
                #print(timestamp)
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
