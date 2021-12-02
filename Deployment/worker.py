import json
from os import path
import numpy as np
import av
from utils import Message, WorkerError, randstr, Stats

# adapted from ARGUSUtils_IO
def load_video(filename):
    '''frame generator
    first item is the number of frames.
    '''
    container = av.open(filename)
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'

    # number of frames
    #yield stream.frames

    for i, frame in enumerate(container.decode(stream)):
        yield frame.to_ndarray(format='gray')

class ArgusWorker:
    def __init__(self, sock):
        self.sock = sock

    def run(self):
        msg = self.sock.recv()
        if msg.type != Message.Type.START:
            raise WorkerError('did not see start msg')
        
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            raise WorkerError('failed to parse start frame')
        
        video_file = data['video_file']

        if not path.exists(video_file):
            raise WorkerError(f'File {video_file} is not accessible!')

        stats = Stats()
        evenodd = 0 # even

        stats.time_start('get frames and inference')
        offset = 0
        for frame in load_video(video_file):
            total = np.sum(frame)
            if total % 2 == evenodd:
                evenodd = 0 # even
            else:
                evenodd = 1 # odd
        stats.time_end('get frames and inference')
        
        result = dict(evenodd=evenodd, stats=stats.todict())
        result_msg = Message(Message.Type.RESULT, json.dumps(result).encode('ascii'))
        self.sock.send(result_msg)