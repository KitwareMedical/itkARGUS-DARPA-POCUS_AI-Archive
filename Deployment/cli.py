import json
from os import path
import ffmpeg, av
import win32file, win32pipe, pywintypes, winerror

from utils import EXIT_FAILURE, Message, PIPE_NAME
from server import WinPipeSock

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

    frame_generator = container.decode(stream)
    # get first frame so we can output info
    first_frame = next(frame_generator).to_ndarray(format='gray')

    yield stream.frames, first_frame.dtype
    yield first_frame
    
    for frame in frame_generator:
        yield frame.to_ndarray(format='gray')

def cli_send_video(video_file, sock):
    if not path.exists(video_file):
        print(f'File {video_file} does not exist')
        return EXIT_FAILURE
    
    # create start_frame msg
    frame_generator = load_video(video_file)
    height, width = shape_video(video_file)
    nframes, dtype = next(frame_generator)
    start_info = dict(
        height=height,
        width=width,
        num_frames=nframes,
        dtype=str(dtype),
    )

    start_msg = Message(Message.Type.START_FRAME, json.dumps(start_info).encode('ascii'))
    sock.send(start_msg)

    # send frame msgs
    for frame in frame_generator:
        frame_msg = Message(Message.Type.FRAME, frame.tobytes())
        sock.send(frame_msg)
    
    result = sock.recv()
    if result.type == Message.Type.RESULT:
        print(f'result: {json.loads(result.data)}')
    elif result.type == Message.Type.ERROR:
        print(f'error: {json.loads(result.data)}')
    else:
        raise Exception('Received message type that is not result nor error')

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
        
        sock = WinPipeSock(handle)
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