import socket
import time


def tcp_test():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_ip = '127.0.0.1'
    server_port = 8051
    client_socket.connect((server_ip, server_port))
    while True:
        client_socket.send("12345".encode('utf-8'))



def tcp_calc():
    # 创建 TCP 套接字
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # 获取当前接收缓冲区大小
    recv_buf_size = s.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
    print(f"默认接收缓冲区大小: {recv_buf_size} 字节")

    # 获取当前发送缓冲区大小
    send_buf_size = s.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
    print(f"默认发送缓冲区大小: {send_buf_size} 字节")

    # 设置接收缓冲区大小为 256KB
tcp_test()