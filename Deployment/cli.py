import argparse
import subprocess
import sys
import json
import time
from os import path
import win32file, win32pipe, pywintypes, winerror

from common import WinPipeSock, Message, Stats, EXIT_FAILURE, PIPE_NAME, LOCK_FILE, LOG_FILE

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

    stats.time_start('inference')
    # create start_frame msg
    start_info = dict(video_file=path.abspath(video_file))

    start_msg = Message(Message.Type.START, json.dumps(start_info).encode('ascii'))
    sock.send(start_msg)

    result = sock.recv()
    stats.time_end('inference')

    if result.type == Message.Type.RESULT:
        srv_res = json.loads(result.data)
        evenodd = 'Yes' if srv_res['evenodd'] else 'No'
        print(f'Sample workload: is the sum of all pixels even or odd?\n\tAnswer = {evenodd}')
        print(f'Total time to read video and produce result: {round(stats.timers["inference"], 2)} seconds')
    elif result.type == Message.Type.ERROR:
        print(f'Error encountered! {json.loads(result.data)}')
    else:
        raise Exception('Received message type that is not result nor error')


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
            print('Trying to connect to service...')
            start_service()
            raise Retry()
        elif code == winerror.ERROR_BROKEN_PIPE:
            print('Server hit an error condition')
            if path.exists(LOG_FILE):
                print(f'Last few lines of server log file ({LOG_FILE}):')
                # not memory efficient, but whatever for now
                with open(LOG_FILE, 'r') as fp:
                    lines = fp.read().strip().split('\n')
                for line in lines[-10:]:
                    print(f'\t{line}')
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
            time.sleep(1)
        except Exception as e:
            print('Fatal error:', e)
            sys.exit(EXIT_FAILURE)

    if path.exists(LOCK_FILE):
        print('The service is in preload phase. Please wait a minute for preload to finalize.')
    else:
        print('The service is not running or exited abnormally.')
        print(f'Please check {LOG_FILE} for details.')
    sys.exit(EXIT_FAILURE)