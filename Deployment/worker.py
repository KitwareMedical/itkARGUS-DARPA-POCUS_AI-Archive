import json
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

        for _ in range(nframes):
            frame_msg = self.sock.recv()
            print('got frame msg', frame_msg)
        
        self.sock.send(Message(Message.Type.RESULT, b'1').tobytes())