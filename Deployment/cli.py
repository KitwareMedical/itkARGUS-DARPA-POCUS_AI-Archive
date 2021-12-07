import argparse
import subprocess
import sys
import json
import time
from os import path
import win32file, win32pipe, pywintypes, winerror

from common import WinPipeSock, Message, Stats, EXIT_FAILURE, PIPE_NAME, EXIT_SUCCESS

class Retry(Exception):
    pass

def prepare_argparser():
    parser = argparse.ArgumentParser(description='ARGUS inference')
    parser.add_argument('video_file', help='video file to analyze.')
    return parser

def start_service():
    subprocess.run(['sc.exe', 'start', 'ARGUS'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def cli_send_video(video_file, sock):
    if not path.exists(video_file):
        print(f'File {video_file} does not exist')
        return EXIT_FAILURE

    stats = Stats()

    # create start_frame msg
    start_info = dict(video_file=path.abspath(video_file))

    start_msg = Message(Message.Type.START, json.dumps(start_info).encode('ascii'))
    sock.send(start_msg)

    stats.time_start('waiting for results')
    result = sock.recv()
    stats.time_end('waiting for results')

    if result.type == Message.Type.RESULT:
        srv_res = json.loads(result.data)
        evenodd = srv_res['evenodd']
        print('yes' if evenodd else 'no')
    elif result.type == Message.Type.ERROR:
        print(f'error: {json.loads(result.data)}')
    else:
        raise Exception('Received message type that is not result nor error')
    
    #print('Self stats:')
    #print(json.dumps(stats.todict(), indent=2))

def main(args):
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
    except pywintypes.error as e:
        code, source, message = e.args
        if code == winerror.ERROR_FILE_NOT_FOUND:
            print('Service not detected; trying to start...')
            start_service()
            raise Retry()
        elif code == winerror.ERROR_BROKEN_PIPE:
            print('broken pipe; server disconnected?')
        elif code == winerror.ERROR_PIPE_BUSY:
            print('server is busy')
        else:
            print('Unknown windows error:', e.args)
        return EXIT_FAILURE
    except Retry:
        raise
    except Exception as e:
        print('cli error:', e)
        return EXIT_FAILURE
    finally:
        if handle:
            win32file.CloseHandle(handle)

if __name__ == '__main__':
    parser = prepare_argparser()
    args = parser.parse_args()
    retries = 0
    while retries < 3:
        try:
            sys.exit(main(args))
        except Retry:
            retries += 1
            time.sleep(0.5)
        except Exception as e:
            print('Fatal error:', e)
            sys.exit(EXIT_FAILURE)