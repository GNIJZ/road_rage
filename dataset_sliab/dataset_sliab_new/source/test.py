import socket
import time

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_ip = '127.0.0.1'
server_port = 8050
client_socket.connect((server_ip, server_port))
while True:
    client_socket.send("message_bytes".encode('utf-8'))
    time.sleep(0.2)