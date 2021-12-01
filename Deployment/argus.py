import argparse
import sys
import logging, logging.handlers
from os import path

from utils import EXIT_SUCCESS
from worker import ArgusWorker
from cli import cli_main
from server import server_main

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

def prepare_argparser():
    parser = argparse.ArgumentParser(description='ARGUS inference')
    parser.add_argument('--server', help='runs server', action='store_true')
    parser.add_argument('video_file', help='video file to analyze.', nargs='?')
    return parser

if __name__ == '__main__':
    parser = prepare_argparser()
    args = parser.parse_args()
    if args.server:
        server_main(ArgusWorker, setup_logger('server'))
    elif args.video_file:
        sys.exit(cli_main(args))
    else:
        parser.print_usage()
        sys.exit(EXIT_SUCCESS)
    # TODO handle service termination signals