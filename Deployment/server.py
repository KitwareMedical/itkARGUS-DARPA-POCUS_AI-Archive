import win32pipe, win32file, pywintypes, win32event, winerror

from utils import Message, WorkerError, PIPE_NAME

INBUF_SIZE = 512 * 1024 * 1024 # 512 MB
OUTBUF_SIZE = 64 * 1024 # 64 KB
MAX_SIZE = 2 * 1024 * 1024 * 1024 # 2 GB

class Sock:
    def recv(self):
        raise NotImplementedError()
    def send(self, data):
        raise NotImplementedError()

class WinPipeSock(Sock):
    def __init__(self, pipe):
        self._pipe = pipe

    def recv(self):
        chunk_size = 64 * 1024
        data = bytearray()
        hr = winerror.ERROR_MORE_DATA
        size = 0
        while hr == winerror.ERROR_MORE_DATA:
            # TODO handle blocking scenario?
            hr, chunk = win32file.ReadFile(self._pipe, chunk_size)
            size += len(chunk)
            if size > MAX_SIZE:
                raise Exception('Exceeded single message max size')
            data.extend(chunk)
        return Message.parse_bytes(bytes(data))
    
    def send(self, msg):
        win32file.WriteFile(self._pipe, msg.tobytes())

class WinPipeServer:
    def __init__(self, WorkerClass, logger):
        self._overlapped = pywintypes.OVERLAPPED()
        self._overlapped.hEvent = win32event.CreateEvent(None, 0, 0, None)
        self._hStop = win32event.CreateEvent(None, 0, 0, None)
        self._quit = False
        self._workerClass = WorkerClass
        self.log = logger
    
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
                sock = WinPipeSock(pipe)
                worker = self._workerClass(sock)
                try:
                    worker.run()
                except WorkerError as e:
                    # TODO send error
                    self.log.error(f'Worker error: {e}')
                break

def server_main(WorkerClass, logger):
    server = WinPipeServer(WorkerClass, logger)
    server.start()

