import argparse
import sys
import logging, logging.handlers
from os import path
import json
import win32pipe, win32file, pywintypes, win32event, winerror
import ffmpeg, av

from utils import Message, WorkerError
from worker import ArgusWorker

PIPE_NAME = r'\\.\pipe\AnatomicRecon-POCUS-AI\inference-server'

INBUF_SIZE = 512 * 1024 * 1024 # 512 MB
OUTBUF_SIZE = 64 * 1024 # 64 KB

EXIT_FAILURE = 1
EXIT_SUCCESS = 0

def setup_logger(name):
    logfile = f'{path.splitext(__file__)[0]}-{name}.log'
    log = logging.getLogger()
    log.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s %(process)d:%(thread)d %(name)s %(levelname)-8s %(message)s')

    stdouthandler = logging.StreamHandler(sys.stdout)
    stdouthandler.setLevel(logging.INFO)
    stdouthandler.setFormatter(formatter)
    log.addHandler(stdouthandler)

    rothandler = logging.handlers.RotatingFileHandler(logfile, maxBytes=8*1024, backupCount=3)
    rothandler.setLevel(logging.INFO)
    rothandler.setFormatter(formatter)
    log.addHandler(rothandler)

    return log

class Sock:
    def recv(self):
        raise NotImplementedError()
    def send(self, data):
        raise NotImplementedError()

class PipeSock(Sock):
    def __init__(self, pipe):
        self._pipe = pipe

    def recv(self):
        chunk_size = 64 * 1024
        data = bytearray()
        hr = winerror.ERROR_MORE_DATA
        while hr == winerror.ERROR_MORE_DATA:
            # TODO handle blocking scenario?
            hr, chunk = win32file.ReadFile(self._pipe, chunk_size)
            data.extend(chunk)
        return Message.parse_bytes(bytes(data))
    
    def send(self, data):
        win32file.WriteFile(self._pipe, data)

class WinPipeServer:
    def __init__(self, WorkerClass):
        self._overlapped = pywintypes.OVERLAPPED()
        self._overlapped.hEvent = win32event.CreateEvent(None, 0, 0, None)
        self._hStop = win32event.CreateEvent(None, 0, 0, None)
        self._quit = False
        self._workerClass = WorkerClass
        self.log = setup_logger('server')
    
    def stop(self):
        self._quit = True
        win32event.SetEvent(self._hStop)
    
    def start(self):
        while not self._quit:
            pipe = win32pipe.CreateNamedPipe(
                PIPE_NAME,
                win32pipe.PIPE_ACCESS_DUPLEX | win32file.FILE_FLAG_OVERLAPPED,
                win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
                1, # 1 instance
                INBUF_SIZE,
                OUTBUF_SIZE,
                0,
                None
            )
            try:
                hr = win32pipe.ConnectNamedPipe(pipe, self._overlapped)
                if hr == winerror.ERROR_PIPE_CONNECTED:
                    win32event.SetEvent(self._overlapped.hEvent)
                self._wait_for_events(pipe)
            except win32pipe.error as e:
                code, source, message = e.args
                # TODO don't handle these errors just yet. need to figure out what to do here
                if code == winerror.ERROR_BROKEN_PIPE:
                    # client likely disconnected.
                    pass
                else:
                    self.log.warn('unknown error:', e)
            finally:
                win32file.CloseHandle(pipe)

    def _wait_for_events(self, pipe ,timeout = 50): # timeout in ms
        while not self._quit:
            rc = win32event.WaitForMultipleObjects((self._hStop, self._overlapped.hEvent), 0, timeout)
            if rc == win32event.WAIT_TIMEOUT:
                continue
            if rc == win32event.WAIT_FAILED:
                self.log.error('Failed to wait!')
                break

            index = rc - win32event.WAIT_OBJECT_0
            if index == 0: # stop signal
                self.stop()
                break
            elif index == 1: # data signal
                sock = PipeSock(pipe)
                worker = self._workerClass(sock)
                try:
                    worker.run()
                except WorkerError as e:
                    # TODO send error
                    self.log.error(f'Worker error: {e}')
                break

def server_main(args):
    server = WinPipeServer(ArgusWorker)
    server.start()

# from ARGUSUtils_IO
def shape_video(filename):
    p = ffmpeg.probe(filename, select_streams='v');
    width = p['streams'][0]['width']
    height = p['streams'][0]['height']
    return height, width

# adapted from ARGUSUtils_IO
def load_video(filename):
    '''frame generator
    first item is the number of frames.
    '''
    container = av.open(filename)
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'

    yield stream.frames
    
    for frame in container.decode(stream):
        yield frame.to_ndarray(format='gray')

def cli_send_video(video_file, sock):
    if not path.exists(video_file):
        print(f'File {video_file} does not exist')
        return EXIT_FAILURE
    
    # create start_frame msg
    frame_generator = load_video(video_file)
    height, width = shape_video(video_file)
    nframes = next(frame_generator)
    start_info = dict(
        height=height,
        width=width,
        num_frames=nframes,
    )

    start_msg = Message(Message.Type.START_FRAME, json.dumps(start_info).encode('ascii'))
    sock.send(start_msg.tobytes())

    # send frame msgs
    for frame in frame_generator:
        frame_msg = Message(Message.Type.FRAME, frame.tobytes())
        sock.send(frame_msg.tobytes())
    
    print(sock.recv().data)

def cli_main(args):
    handle = None
    try:
        handle = win32file.CreateFile(
            PIPE_NAME,
            win32file.GENERIC_READ | win32file.GENERIC_WRITE,
            0,
            None,
            win32file.OPEN_EXISTING,
            0,
            None
        )
        res = win32pipe.SetNamedPipeHandleState(handle, win32pipe.PIPE_READMODE_MESSAGE, None, None)
        if res == 0:
            print(f'SetNamedPipeHandleState return code: {res}')
            return
        
        sock = PipeSock(handle)
        return cli_send_video(args.video_file, sock)
        #count = 0
        #while count < 10:
        #    win32file.WriteFile(handle, Message(Message.Type.START_FRAME, b'1' * 1024).tobytes())
        #    count += 1
    except pywintypes.error as e:
        code, source, message = e.args
        if code == winerror.ERROR_FILE_NOT_FOUND:
            print('no pipe')
        elif code == winerror.ERROR_BROKEN_PIPE:
            print('broken pipe; server disconnected?')
        elif code == winerror.ERROR_PIPE_BUSY:
            print('server is busy')
        return EXIT_FAILURE
    finally:
        if handle:
            win32file.CloseHandle(handle)

def prepare_argparser():
    parser = argparse.ArgumentParser(description='ARGUS inference')
    parser.add_argument('--server', help='runs server', action='store_true')
    parser.add_argument('video_file', help='video file to analyze.', nargs='?')
    return parser

if __name__ == '__main__':
    parser = prepare_argparser()
    args = parser.parse_args()
    if args.server:
        server_main(args)
    elif args.video_file:
        sys.exit(cli_main(args))
    else:
        parser.print_usage()
        sys.exit(EXIT_SUCCESS)
    # TODO handle service termination signals