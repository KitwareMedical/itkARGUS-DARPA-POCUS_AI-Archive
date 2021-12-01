from enum import Enum

PIPE_NAME = r'\\.\pipe\AnatomicRecon-POCUS-AI\inference-server'

EXIT_FAILURE = 1
EXIT_SUCCESS = 0
class Message:
    # message type is a single byte
    class Type(Enum):
        # messages from cli
        START = 0x1
        FRAMES_WRITTEN = 0x2
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