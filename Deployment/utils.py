from enum import Enum
import string
import random
import time

PIPE_NAME = r'\\.\pipe\AnatomicRecon-POCUS-AI\inference-server'

EXIT_FAILURE = 1
EXIT_SUCCESS = 0
class Message:
    # message type is a single byte
    class Type(Enum):
        # messages from cli
        START = 0x1
        # messages from server
        RESULT = 0x81
        ERROR = 0x82
        SHM = 0x83

        def tobyte(self):
            return self.value.to_bytes(1, 'big')
        
        @classmethod
        def frombyte(cls, byte):
            return cls(int.from_bytes(byte, 'big'))

    def __init__(self, mtype, data):
        '''mtype can either be an integer or a Message.Type'''
        self.type = Message.Type(mtype)
        self.data = data
    
    def tobytes(self):
        return self.type.tobyte() + self.data
    
    @classmethod
    def parse_bytes(cls, data):
        return cls(Message.Type.frombyte(data[:1]), data[1:])

class WorkerError(Exception):
    pass

def randstr(length):
    size = len(string.ascii_letters)
    return ''.join(string.ascii_letters[random.randint(0, size-1)] for _ in range(length))

class Stats:
    timers = dict()
    _running_timers = dict()

    def time_start(self, name):
        start = time.time()
        self._running_timers[name] = start

    def time_end(self, name):
        end = time.time()
        if name in self._running_timers:
            self.timers[name] = end - self._running_timers[name]
            del self._running_timers[name]
    
    def todict(self):
        return dict(timers=self.timers)