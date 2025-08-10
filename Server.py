import socket
import struct
import cv2
import numpy as np
import torch
import json
from ultralytics import YOLO






# Model yükleme
model = YOLO('best.pt')
#results = model("img3.jpg")  # size param optional (inference boyutu)
#print(results[0].boxes.xyxy.tolist())



host = "127.0.0.1"
port = 5000

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, port))
server_socket.listen(1)
print("python server is started")

conn, addr = server_socket.accept()
print(f"{addr} is connected.")

def recvall(sock, length):
    """Belirtilen uzunlukta byte okuyana kadar bekler."""
    data = b''
    while len(data) < length:
        packet = sock.recv(length-len(data))
        if not packet:
            return None
        data += packet
    return data

while True:
    lengthbuf = recvall(conn, 4)
    if not lengthbuf:
        break
    length = struct.unpack('<I', lengthbuf)[0]
    
    img_bytes = recvall(conn, length)
    if img_bytes is None:
        break
    print("Image is received")
    
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # model(img) wrapper, hem file path hem PIL hem numpy array kabul eder
    results = model(img)  # size param optional (inference boyutu)
    result = results[0]
    data_message = {"Detections": []}
    
    for i in range(len(result.boxes)):
        box = result.boxes.xyxy[i].tolist()
        
        data_message["Detections"].append({
            "box": box
        })
        
    json_data = json.dumps(data_message)
    conn.sendall(json_data.encode('utf-8'))
    
    """cv2.imshow("İmage which received from Unity", img)
    if cv2.waitKey(1) == 27:  # ESC
        break"""
    

"""while True:
    data = conn.recv(1024).decode()
    if not data:
        print("Disconnected")
        break
    print("received from unity: ", data)
    conn.sendall("Hello Unity".encode())"""

conn.close()
cv2.destroyAllWindows()