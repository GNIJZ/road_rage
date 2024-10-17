import socket

# 创建 TCP 套接字
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 获取当前接收缓冲区大小
recv_buf_size = s.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
print(f"默认接收缓冲区大小: {recv_buf_size} 字节")

# 获取当前发送缓冲区大小
send_buf_size = s.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF)
print(f"默认发送缓冲区大小: {send_buf_size} 字节")

# 设置接收缓冲区大小为 256KB

