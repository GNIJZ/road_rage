import socket

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_ip = '127.0.0.1'
server_port = 8051
client_socket.connect((server_ip, server_port))
while True:
    client_socket.send("12345".encode('utf-8'))