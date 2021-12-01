import json
import numpy as np
from utils import Message, WorkerError

class ArgusWorker:
    def __init__(self, sock):
        self.sock = sock

    def run(self):
        msg = self.sock.recv()
        if msg.type != Message.Type.START_FRAME:
            raise WorkerError('did not see start frame')
        
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            raise WorkerError('failed to parse start frame')
        
        nframes = data['num_frames']
        width = data['width']
        height = data['height']
        dtype = data['dtype']

        evenodd = 0 # even

        for _ in range(nframes):
            frame_msg = self.sock.recv()
            if frame_msg.type != Message.Type.FRAME:
                raise WorkerError('expected a frame')
            frame = np.frombuffer(frame_msg.data, dtype=dtype).reshape((height, width))
            total = np.sum(frame)
            if total % 2 == evenodd:
                evenodd = 0 # even
            else:
                evenodd = 1 # odd
        
        result = dict(evenodd=evenodd)
        result_msg = Message(Message.Type.RESULT, json.dumps(result).encode('ascii'))
        self.sock.send(result_msg)