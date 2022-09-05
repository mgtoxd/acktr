import sys
import threading

import grpc
import time
from concurrent import futures
import landlord2Cloud_pb2_grpc, landlord2Cloud_pb2
from Base import BaseChoose
from Core import CoreChoose
from a2c_ppo_acktr.arguments import get_args
_ONE_DAY_IN_SECONDS = 60 * 60 * 24
_HOST = '0.0.0.0'
_PORT = '5000'


# 节点数量*文件数量*延迟
class FormatDataA(landlord2Cloud_pb2_grpc.ACKTRServicer):
    def __init__(self, b: BaseChoose):
        super().__init__()
        self.BaseChoose = b

    def GetDem(self, request_iterator, context):
        self.BaseChoose.reset()
        for item in request_iterator:
            self.BaseChoose.append(item=item)
        try:
            h = self.BaseChoose.getHighest()
        except Exception:
            print(sys.exc_info())

        print("ooutL:", h)
        return landlord2Cloud_pb2.Hash(hash=h)


def serveA(i, b: BaseChoose):
    # 定义服务器并设置最大连接数,concurrent.futures是一个并发库，类似于线程池的概念
    grpcServer = grpc.server(futures.ThreadPoolExecutor(max_workers=4))  # 创建一个服务器
    landlord2Cloud_pb2_grpc.add_ACKTRServicer_to_server(FormatDataA(b), grpcServer)  # 在服务器中添加派生的接口服务（自己实现了处理函数）
    print(str(5000 + i))
    grpcServer.add_insecure_port(_HOST + ':' + str(5000 + i))  # 添加监听端口
    grpcServer.start()  # 启动服务器
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        grpcServer.stop(0)  # 关闭服务器


if __name__ == '__main__':
    sizeMap = {}
    channel = grpc.insecure_channel('192.168.201.241:8888')
    stub = landlord2Cloud_pb2_grpc.CloudStub(channel)
    rs = stub.GetFileList(landlord2Cloud_pb2.File())
    for i in rs:
        print(i)
        sizeMap[i.hash] = i.size
    print(sizeMap)
    for i in range(3, 15):
        b = CoreChoose(sizeMap=sizeMap,args=get_args())
        t = threading.Thread(target=serveA, args=(i, b))
        t.start()
    while True:
        time.sleep(500000)
