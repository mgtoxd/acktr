# BaseChoose 选择延迟最高的
from landlord2Cloud_pb2 import DemandInfoList


class BaseChoose:
    def __init__(self, sizeMap):
        self.sizeMap = sizeMap
        self.delayMap = {}
        pass

    def append(self, item: DemandInfoList):
        for info in item.info:
            preDelay = (info.count * item.net_delay) + (item.cal_delay * info.count * self.sizeMap[info.hash])
            if self.delayMap.__contains__(info.hash):
                self.delayMap[info.hash]['delay'] += (info.delay - preDelay)
                self.delayMap[info.hash]['count'] += info.count
            else:
                self.delayMap[info.hash] = {'delay': (info.delay - preDelay), 'count': info.count}

    def reset(self):
        self.delayMap = {}

    def getHighest(self):

        maxKey = -1
        maxV = 0
        for key, value in self.delayMap.items():
            print(key,value)
            if value['delay'] / (value['count'] * self.sizeMap[key]) > maxV:
                maxV = value['delay'] / (value['count'] * self.sizeMap[key])
                maxKey = key
        return maxKey
