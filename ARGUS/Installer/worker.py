import sys
import os
from os import path
import json
import numpy as np
import gc

# pyinstaller: import before itk, since itk.support imports torch
# and having itk import torch causes incomplete loading of torch._C
# for some reason...
import torch

import itk
itk.force_load()

from common import Message, WorkerError, randstr, Stats

def is_bundled():
    return getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')

def get_ARGUS_dir():
    if is_bundled():
        return path.join(sys._MEIPASS, 'ARGUS')
    return path.join('../..', 'ARGUS')

def preload_itk_argus():
    # avoids "ITK not compiled with TBB" errors
    # this just does an instantiation prior to actuallly using it
    T = itk.Image[itk.F,2]
    itk.itkARGUS.ResampleImageUsingMapFilter[T,T].New()

preload_itk_argus()

# load argus stuff after ITK
sys.path.append(get_ARGUS_dir())
from ARGUS_app_ai import ARGUS_app_ai

class ArgusWorker:
    def __init__(self, sock, log):
        self.sock = sock
        self.log = log
        self.app_ai = ARGUS_app_ai(argus_dir=get_ARGUS_dir())

    def run(self):
        stats = Stats()
        gc.disable()
        self.handle_request()

        with stats.time('manual gc'):
            gc.collect()
            gc.enable()
        
        self.log.info(f'Manual GC overhead: {stats.timers["manual gc"]["elapsed"]}')
    
    def handle_request(self):
        stats = Stats()

        msg = self.sock.recv()
        if msg.type != Message.Type.START:
            raise WorkerError('did not see start msg')
        
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            raise WorkerError('failed to parse start frame')
        
        video_file = data['video_file']
        source = data['source']
        task = data.get('task', None)
        device_num = data.get('device_num', None)
        if device_num != None and device_num.isdigit():
            device_num = int(device_num)

        try:
            if not path.exists(video_file):
                raise Exception(f'File {video_file} is not accessible!')

            inf_result = self.app_ai.predict(video_file, stats=stats, task=task, source=source, device_num=device_num)
        except Exception as e:
            self.log.exception(e)
            error_msg = Message(Message.Type.ERROR, json.dumps(str(e)).encode('ascii'))
            self.sock.send(error_msg)
            return

        if inf_result != None:
            result = dict(
                filename=video_file,
                task_name=inf_result['task_name'],
                source=source,
                device_num=device_num,
                prediction=inf_result['decision'],
                stats=stats.todict(),
                video_length=inf_result['video_length'],
                task_confidence_PTX=inf_result['task_confidence_PTX'],
                task_confidence_PNB=inf_result['task_confidence_PNB'],
                task_confidence_ONSD=inf_result['task_confidence_ONSD'],
                task_confidence_ETT=inf_result['task_confidence_ETT'],
                decision_confidence_0=inf_result['decision_confidence_0'],
                decision_confidence_1=inf_result['decision_confidence_1'],
            )
            result_msg = Message(Message.Type.RESULT, json.dumps(result).encode('ascii'))
            self.sock.send(result_msg)
