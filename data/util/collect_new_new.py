import csv

import cv2
import keyboard
import pyaudio
import wave
import os
import socket
import threading
import time
import csv
import queue


class CollectImage:
    def __init__(self, frame_rate, dir, save):
        self.frame_rate = frame_rate
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.save_event = threading.Event()
        self.frames = []
        self.save = save
        self.image_queue = queue.Queue()
        self.timestamp=''
        self.Image_get=False
        if save:
            os.makedirs(os.path.join(dir, "vision"), exist_ok=True)
            self.save_dir = os.path.join(dir, "vision")

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.timestamp = time.strftime('%Y%m%d%H%M%S')
                self.image_queue.put((self.timestamp, frame))
                if self.image_queue.qsize() >= 2 and self.save_event.is_set():
                    first_frame = self.image_queue.queue[0]
                    mid_index = self.image_queue.qsize() // 2
                    last_frame = self.image_queue.queue[mid_index]
                    self.frames = [first_frame, last_frame]
                    self.image_queue.queue.clear()  # 清空队列
                    if self.save:
                        self.save_data(self.frames)
                    else:
                        self.Image_get=True
                    self.save_event.clear()

    def save_data(self, frames):
        for index, (timestamp, frame) in enumerate(frames):
            file_name = f"{timestamp}_{index}.jpg"
            file_path = os.path.join(self.save_dir, file_name)
            cv2.imwrite(file_path, frame)
        self.frames.clear()

    def getdata(self):
        return self.frames

    def stop(self):
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()


class AudioCollector:
    def __init__(self, dir, save):
        os.makedirs(os.path.join(dir, "audio"), exist_ok=True)
        self.save_event = threading.Event()
        self.dir = os.path.join(dir, "audio")
        self.chunk = 1600
        self.sample_format = pyaudio.paInt16
        self.channels = 2
        self.fs = 16000
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.sample_format, channels=self.channels,
                                  rate=self.fs, frames_per_buffer=self.chunk, input=True)
        self.frames_queue = queue.Queue()
        self.running = True
        self.save = save
        self.frames=[]
        self.Audio_get=False
    def run(self):
        while self.running:
            try:
                data = self.stream.read(self.chunk, exception_on_overflow=False)
                timestamp = time.strftime('%Y%m%d%H%M%S')
                self.frames_queue.put((timestamp, data))
                if self.save:
                    pass
                else:
                    if self.save_event.is_set():
                        while not self.frames_queue.empty():
                            self.frames.append(self.frames_queue.get())
                    self.save_event.clear()
                    self.Audio_get=True
            except IOError as e:
                print(f"Error reading stream: {e}")
                break

    def save_data(self):
        if self.frames_queue.empty():
            return
        first_timestamp, _ = self.frames_queue.queue[0]
        output_file = os.path.join(self.dir, f"{first_timestamp}.wav")
        with wave.open(output_file, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.p.get_sample_size(self.sample_format))
            wf.setframerate(self.fs)
            while not self.frames_queue.empty():
                timestamp, data = self.frames_queue.get()
                wf.writeframes(data)

    def getdata(self):
        data = []
        while not self.frames_queue.empty():
            data.append(self.frames_queue.get())
        return data

    def stop(self):
        if self.save:
            self.save_data()
        self.running = False
        self.save_event.clear()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


class CollectSilab:
    def __init__(self, ip, port, frame_rate, dir, save):
        self.frame_rate = frame_rate
        self.sensor_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sensor_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 256 * 1024)
        self.sensor_socket.bind((ip, port))
        self.sensor_socket.listen(1)
        print(f"Server listening on {ip}:{port}")
        self.conn, addr = self.sensor_socket.accept()
        print(f"Connection accepted from {addr}")
        self.running = True
        self.save_event = threading.Event()
        self.save = save
        self.buffered_data = []
        self.data_queue = queue.Queue()
        self.Silab_get=False
        if save:
            os.makedirs(os.path.join(dir, "silab"), exist_ok=True)
            self.save_dir = os.path.join(dir, "silab")
            self.csv_file_path = os.path.join(self.save_dir, 'silab_data.csv')
            self.writer = open(self.csv_file_path, mode='w', newline='', encoding="utf-8")
            self.csv_writer = csv.writer(self.writer)
            self.csv_writer.writerow(['Timestamp', '刹车', '油门', '方向盘'])

    def run(self):
        while self.running:
            try:
                chunk = self.conn.recv(14).decode("utf-8")
                self.timestamp = time.strftime('%Y%m%d%H%M%S')
                if chunk:
                    self.data_queue.put((self.timestamp, chunk))
                if self.save_event.is_set():
                    # 处理保存逻辑
                    while not self.data_queue.empty():
                        data = self.data_queue.get()
                        self.buffered_data.append(data)
                    if len(self.buffered_data) >=10:
                        self.buffered_data = [self.buffered_data[i * len(self.buffered_data) // 10] for i in range(0, 10)]
                    if self.save:
                        if len(self.buffered_data) >= 10:
                            self.csv_writer.writerows(self.buffered_data)
                            self.buffered_data.clear()
                    else:
                        self.Silab_get=True
                        pass
                        # 在不保存时，直接获取数据
                        # self.getdata()
                    self.save_event.clear()
            except socket.error as e:
                print(f"Socket error: {e}")
                break
            except Exception as e:
                print(f"Error receiving data: {e}")
                break

    def getdata(self):
        data = self.buffered_data.copy()
        self.buffered_data.clear()
        return data
        # while not self.data_queue.empty():
        #     data.append(self.data_queue.get())
        # return data

    def stop(self):
        self.running = False
        if self.save and self.buffered_data:
            self.csv_writer.writerows(self.buffered_data)
            self.writer.close()
        self.conn.close()
        self.sensor_socket.close()


def start_threads(collectImage, collectSilab, collectAudio):
    """启动线程"""
    image_thread = threading.Thread(target=collectImage.run)
    silab_thread = threading.Thread(target=collectSilab.run)
    audio_thread = threading.Thread(target=collectAudio.run)

    image_thread.start()
    silab_thread.start()
    audio_thread.start()

    return image_thread, silab_thread, audio_thread


def process_data(collectImage, collectSilab, collectAudio):
    """处理采集的数据"""
    while True:
        if collectImage.Image_get and collectSilab.Silab_get and collectAudio.Audio_get:
            data1 = collectImage.getdata()
            data2 = collectSilab.getdata()
            data3 = collectAudio.getdata()

            if len(data1) > 0:
                timestamp, frame = data1[0]
                print("图像个数：", len(data1), "  图片时间戳和维度", timestamp, "  ", frame.shape)
            else:
                print("没有图像数据")
            if len(data2) > 0:
                timestamp, silab_data = data2[0]
                print("silab个数：", len(data2), "  silab时间戳和维度", timestamp, "  ", silab_data)
            else:
                print("没有silab数据")
            if len(data3) > 0:
                timestamp, audio_data = data3[0]
                print("音频个数：", len(data3), "  音频时间戳和维度", timestamp, "  ", audio_data)
            else:
                print("没有音频数据")
            print("\n")
            collectImage.Image_get=collectSilab.Silab_get=collectAudio.Audio_get=False
            return


def main():
    _Save = False
    directory_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)), "dataset_silab",
                                  "data", "gnij02")
    os.makedirs(directory_path, exist_ok=True)

    collectImage = CollectImage(2, dir=directory_path, save=_Save)
    collectSilab = CollectSilab("127.0.0.1", 8051, 10, dir=directory_path, save=_Save)
    collectAudio = AudioCollector(dir=directory_path, save=_Save)

    # 启动线程
    image_thread, silab_thread, audio_thread = start_threads(collectImage, collectSilab, collectAudio)
    last_printed_second = int(time.time())

    try:
        while True:
            current_time = int(time.time())
            if current_time > last_printed_second:
                if (collectImage.image_queue.qsize() < 2) or (collectSilab.data_queue.qsize() < 10):
                    # 清空数据
                    collectImage.image_queue.queue.clear()
                    collectSilab.data_queue.queue.clear()  # 清空队列
                    collectAudio.frames_queue.queue.clear()  # 清空音频数据队列
                    last_printed_second = current_time
                    print("数据不足，等待下一轮")
                    continue  # 继续循环，等待下一秒

                # 设置事件以允许保存数据
                collectImage.save_event.set()
                collectSilab.save_event.set()
                collectAudio.save_event.set()

                if not _Save:
                    process_data(collectImage, collectSilab, collectAudio)
                print(time.strftime('%Y%m%d%H%M%S'))
                last_printed_second = current_time  # 更新已打印的秒数

            if keyboard.is_pressed('esc'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        collectImage.stop()
        collectAudio.stop()
        collectSilab.stop()
        image_thread.join()
        audio_thread.join()
        silab_thread.join()
        print("数据采集已停止")


if __name__ == '__main__':
    main()
