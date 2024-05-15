#服务端,识别算法 TCP协议
from socketfunc import *
import struct
import threading
import time
from detection_1 import *
import os
from datetime import datetime
import cv2
from tobaccoImageEnhence_release import *
#日志记录文件夹
def log_dir_init(dir_path= "./logs/ai_logs/"):
       # 目录路径
        #dir_path = "./logs/ai_logs"
        # 检查目录是否存在
        if not os.path.exists(dir_path):
                # 创建目录
                os.makedirs(dir_path)
#记录日记
def log_record(dir_path,texts):
       # 获取当前日期和时间
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        # 构造文件名和文本内容
        filename = f"{date_str}.txt"
        text = f"curtime:{time_str}\n"
        filename = os.path.join(os.path.dirname(dir_path), filename)
        # 检查文件是否存在
        if os.path.exists(filename):
                # 追加文本到文件
                with open(filename, "a") as file:
                        file.write(text)
                        for log in texts:
                                file.write(log)
        else:
                # 创建新文件并保存文本
                with open(filename, "w") as file:
                        file.write(text)
                        for log in texts:
                                file.write(log)

dir_path= r"./logs/ai_logs/"
log_dir_init(dir_path= dir_path)      

def send_message(message,remote_host, remote_port):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((remote_host, remote_port))
    bytes_sent = client_socket.sendall(message.encode('utf-8'))
    # data = client_socket.recv(1024)
    # print("收到服务端消息：", data.decode())
    client_socket.close()
    return bytes_sent
#socket   
host = '127.0.0.1'  # 主机名或IP地址
recvport = 6021  # 端口号
sendport = 6021
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host, recvport))
server_socket.listen(5)
# print("服务器启动，监听本地端口：%d" % recvport)



#初始化ai模型
model = model_init(r'./pth/best_model.pth')
#调用识别模型，推理
testclass,ppid = detection('./test.bmp', '0',model)
print("test class:",testclass) 
print("model loaded successfully")
#创建显示线程
threadx = threading.Thread(target=showTobaccoImageThread)    
# 启动线程
threadx.start()  

while True:
        # 接收数据
        logtext = []
        #data = receive_data(recvport)
        client_socket, addr = server_socket.accept()
        data = client_socket.recv(1024)
        start_time = time.time()
        msg = data.decode('utf-8')
        logtext.append(f"\t recvmsg is {msg}\n")        
        separated_list = msg.split("|")
        model_path = separated_list[1]
        img_path = separated_list[0]
        uuid =  separated_list[3]
        print("model_path",model_path,"\t",'img_path',img_path,'\t','uuid:',uuid)
        last_chars = img_path[-8:]
        print("side_str",last_chars)
        if (img_path[-8:]=="side.bmp"):                
                udp_data = "|".join(["0", uuid])
                logtext.append(f'\t side图片不识别，直接输出数据,{udp_data}\n')
                print('side图片不识别，直接输出数据', udp_data)
                # send_message(udp_data, host, sendport)
                client_socket.sendall(udp_data.encode('utf-8'))
        else:
            # 图像识别
            #调用识别模型，推理
            output_class,picture_id = detection(img_path, uuid,model)
            print([output_class,picture_id])
            udp_data = "|".join([output_class, picture_id])
            print('识别结果', udp_data)
            logtext.append(f'\t recognize result , class and id is {udp_data}\n')
            #保存增强图像
        #     print(img_path)
            saveTobaccoImageThread(img_path)  
            # 发送数据
            bytes_sent = 0
        #     bytes_sent = send_message(udp_data, host, sendport)
            bytes_sent =client_socket.sendall(udp_data.encode('utf-8'))
            logtext.append(f'\t send_data success!!! \n bytes_sent is {bytes_sent}\n')
        #     print(img_path)
            #saveImage(img_path)            
            end_time = time.time()
            total_time = end_time - start_time
            print("total time:",total_time)
            logtext.append(f'\t total time: {total_time}\n')
        log_record(dir_path=dir_path,texts=logtext)



cv2.destroyAllWindows()
