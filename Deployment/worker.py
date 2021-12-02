import json

import numpy as np
from multiprocessing import shared_memory
import win32event, win32api
from utils import Message, WorkerError, randstr, Stats

def InterruptableWaitForSingleObject(*args):
    while True:
        rc = win32event.WaitForSingleObject(*args)
        if rc == win32event.WAIT_OBJECT_0:
            break

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
        frame_sem = None
        try:
            sem_name = f'Global\\ARGUS-{randstr(20)}'
            # TODO how to detect failure in CreateSemaphore
            frame_sem = win32event.CreateSemaphore(None, 0, nframes, sem_name)
            shm = shared_memory.SharedMemory(create=True, size=totalmem)

            shm_msg = Message(
                Message.Type.SHM,
                json.dumps(dict(shm_name=shm.name, sem_name=sem_name)).encode('ascii')
            )
            self.sock.send(shm_msg)

            stats = Stats()
            evenodd = 0 # even

            stats.time_start('get frames and inference')
            offset = 0
            for _ in range(nframes):
                # wait for available frames
                InterruptableWaitForSingleObject(frame_sem, 1) # wait 1ms

                frame = np.ndarray((height, width), dtype=dtype, buffer=shm.buf, offset=offset)
                offset += frame_byte_size

                total = np.sum(frame)
                if total % 2 == evenodd:
                    evenodd = 0 # even
                else:
                    evenodd = 1 # odd
            stats.time_end('get frames and inference')
            
            result = dict(evenodd=evenodd, stats=stats.todict())
            result_msg = Message(Message.Type.RESULT, json.dumps(result).encode('ascii'))
            self.sock.send(result_msg)
        finally:
            if frame_sem:
                win32api.CloseHandle(frame_sem)
            if shm:
                shm.close()
                # we "own" the shm, so unlink it
                shm.unlink()