import json
import numpy as np
from multiprocessing import shared_memory
from utils import Message, WorkerError

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
        
        nframes = data['num_frames']
        width = data['width']
        height = data['height']
        dtype = data['dtype']

        frame_byte_size = width * height * np.dtype(dtype).itemsize
        totalmem = nframes * frame_byte_size

        shm = None
        try:
            shm = shared_memory.SharedMemory(create=True, size=totalmem)
            shm_msg = Message(
                Message.Type.SHM,
                json.dumps(dict(shm_name=shm.name)).encode('ascii')
            )
            self.sock.send(shm_msg)

            # wait for client to write all of the frames
            written_msg = self.sock.recv()
            if written_msg.type != Message.Type.FRAMES_WRITTEN:
                raise WorkerError('did not see frames_written msg')

            evenodd = 0 # even

            offset = 0
            for _ in range(nframes):
                frame = np.ndarray((height, width), dtype=dtype, buffer=shm.buf, offset=offset)
                offset += frame_byte_size

                total = np.sum(frame)
                if total % 2 == evenodd:
                    evenodd = 0 # even
                else:
                    evenodd = 1 # odd
            
            result = dict(evenodd=evenodd)
            result_msg = Message(Message.Type.RESULT, json.dumps(result).encode('ascii'))
            self.sock.send(result_msg)
        finally:
            if shm:
                shm.close()
                # we "own" the shm, so unlink it
                shm.unlink()