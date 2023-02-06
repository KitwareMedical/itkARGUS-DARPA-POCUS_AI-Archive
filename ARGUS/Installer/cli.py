import argparse
import subprocess
import sys
import json
import time
import csv
import traceback
from os import path
from glob import glob
import win32file, win32pipe, pywintypes, winerror

from common import WinPipeSock, Message, EXIT_FAILURE, PIPE_NAME, LOCK_FILE, LOG_FILE, EXIT_SUCCESS

tasks = [ "PTX", "PNB", "ONSD", "ETT" ]
sources = [ "Sonosite", "Butterfly", "Clarius" ]

class Retry(Exception):
    pass

def prepare_argparser():
    parser = argparse.ArgumentParser(description='ARGUS inference')
    parser.add_argument('-f', '--file',
                        help='Video file to analyze. REQUIRED: -f or -d')
    parser.add_argument('-d', '--directory',
                        help='Directory of video (*.mp4 and *.mov) files'
                             ' to be analyzed. REQUIRED: -f or -d')
    parser.add_argument('-s', '--source', 
                        help='Specify ultrasound probe type:'
                             ' Butterfly, Sonosite, Clarius.  REQUIRED')
    parser.add_argument('-g', '--gpu',
                        help='Accelerate using the specified GPU.')
    parser.add_argument('-t', '--task', 
                        help='Specify task: PTX, PNB, ONSD, ETT.'
                             ' This will override the automatic task'
                             ' determination AI.')
    parser.add_argument('-D', '--Debug', action='store_true',
                        help='Enable debugging.')
    return parser

def start_service():
    subprocess.run(['sc.exe', 'start', 'ARGUS'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def formatHHMMSS(secs=None):
    if secs is None:
        secs = time.time()
    msecs = int(1000 * (secs - int(secs)))
    return f'{time.strftime("%H:%M:%S", time.gmtime(secs))}:{msecs}'

def dbg(*args, **kwargs):
    print(f'DEBUG [{formatHHMMSS()}]:', *args, **kwargs)

def write_result(video_file, result, debug=False):
    result_filename = path.join(
        path.dirname(path.abspath(video_file)),
        f'{path.splitext(path.basename(video_file))[0]}.csv'
    )

    stats = result['stats']
    timers = stats['timers']
    prediction = result['prediction']

    task_name = result['task_name']
    source = result['source']
    device_num = result['device_num']

    video_length = result['video_length']

    time_reading=round(timers['Read Video']['elapsed'], 3)
    time_preprocessing=round(timers['Preprocess Video']['elapsed'], 3)
    time_processing=round(timers['Process Video']['elapsed'], 3)
    time_total=round(timers['all']['elapsed'], 3)
    time_total_with_parallelism = time_total - min(time_processing,
                                                   video_length)

    task_confidence_PTX = result['task_confidence_PTX']
    task_confidence_PNB = result['task_confidence_PNB']
    task_confidence_ONSD = result['task_confidence_ONSD']
    task_confidence_ETT = result['task_confidence_ETT']
    decision_confidence_0 = result['decision_confidence_0']
    decision_confidence_1 = result['decision_confidence_1']

    csv_data = dict(
        filename=result_filename,
        task=task_name,
        prediction=prediction,
        time_reading_the_video=time_reading,
        time_preprocessing_the_video=time_preprocessing,
        time_processing_the_video=time_processing,
        time_total=time_total,
        time_total_with_parallelism=time_total_with_parallelism,
        video_length=video_length,
        task_confidence_PTX=task_confidence_PTX,
        task_confidence_PNB=task_confidence_PNB,
        task_confidence_ONSD=task_confidence_ONSD,
        task_confidence_ETT=task_confidence_ETT,
        source=source,
        device_num=device_num,
        decision_confidence_0=decision_confidence_0,
        decision_confidence_1=decision_confidence_1,
    )

    with open(result_filename, 'w', newline='') as fp:
        fieldnames = list(csv_data.keys())
        writer = csv.DictWriter(
            fp,
            fieldnames=fieldnames,
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        writer.writerow(csv_data)

    print(f'File: {video_file}')
    print(f'   Task: {task_name}')
    print(f'   Prediction: {prediction}')
    print(f'      Confidence Measure 0: {decision_confidence_0}')
    print(f'      Confidence Measure 1: {decision_confidence_1}')

def cli_send_video(video_file, sock, task=None, source=None, device_num=None, debug=False):
    if not path.exists(video_file):
        print(f'File {video_file} does not exist')
        return None

    # create start_frame msg
    start_info = dict(video_file=path.abspath(video_file),
                      task=task,
                      source=source,
                      debug=debug)

    if debug:
        dbg('Sending start message...')

    start_msg = Message(Message.Type.START, json.dumps(start_info).encode('ascii'))
    sock.send(start_msg)

    if debug:
        dbg('...start message sent.')
        dbg('Waiting on result message...')

    result = sock.recv()

    if debug:
        dbg('...result message received.')

    if result.type == Message.Type.RESULT:
        return json.loads(result.data)
    elif result.type == Message.Type.ERROR:
        print(f'Error encountered! {json.loads(result.data)}')
        return None
    else:
        raise Exception('Received message type that is not result nor error')


def main(args):
    debug = args.Debug
    
    task = None
    if args.task != None:
        if args.task in tasks:
            task = args.task
        else:
            print(f"ERROR: Task {args.task} not defined.")
            print(f"    -t {task}")
            print(f"    Use the -h option for details.")
            return

    source = None
    if args.source == None:
        print(f"ERROR: Source required: -s {sources}")
        return
    if args.source in sources:
        source = args.source
    else:
        print(f"ERROR: Source {args.source} not defined.")
        print(f"    -s {sources}")
        print(f"    Use the -h option for details.")
        return
    
    device_num = None
    if args.gpu != None:
        if args.gpu.isdigit():
            device_num = int(args.gpu)
        else:
            print(f"ERROR: Device number {args.gpu} not defined.")
            print(f"    Use the -h option for details.")
            return
    
    handle = None
    try:
        if args.file != None:
            handle = win32file.CreateFile(
                PIPE_NAME,
                win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                0,
                None,
                win32file.OPEN_EXISTING,
                0,
                None
            )
            res = win32pipe.SetNamedPipeHandleState(
                handle,
                win32pipe.PIPE_READMODE_MESSAGE,
                None,
                None
            )
            if res == 0:
                print(f'SetNamedPipeHandleState return code: {res}')
                return
            sock = WinPipeSock(handle)
            result = cli_send_video(args.file,
                                    sock,
                                    task=task,
                                    source=source,
                                    device_num=device_num,
                                    debug=debug)
            if result:
                write_result(args.file, result, debug=debug)
                if handle:
                    win32file.CloseHandle(handle)
                return EXIT_SUCCESS
            return EXIT_FAILURE
        elif args.directory != None:
            files = sorted(glob(os.path.join(args.directory, "*.m??")))
            for vidfile in files:
                handle = win32file.CreateFile(
                    PIPE_NAME,
                    win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                    0,
                    None,
                    win32file.OPEN_EXISTING,
                    0,
                    None
                )
                res = win32pipe.SetNamedPipeHandleState(
                    handle,
                    win32pipe.PIPE_READMODE_MESSAGE,
                    None,
                    None
                )
                if res == 0:
                    print(f'SetNamedPipeHandleState return code: {res}')
                    return
                sock = WinPipeSock(handle)
                result = cli_send_video(vidfile,
                                        sock,
                                        task=task,
                                        source=source,
                                        device_num=device_num,
                                        debug=debug)
                if result:
                    write_result(vidfile, result, debug=debug)
                if handle:
                    win32file.CloseHandle(handle)
                time.sleep(5)

            return EXIT_SUCCESS
        else:
            print('Please specify -f <filename> or -d <directory>.')
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
            raise Retry()
        else:
            print('Unknown windows error:', e.args)
        return EXIT_FAILURE
    except Retry:
        raise
    except Exception as e:
        print('cli error:')
        traceback.print_exc()
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
