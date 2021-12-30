import sys
import os
from os import path
import json
import numpy as np

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
    return path.join('..', 'ARGUS')

def get_model_dir():
    return path.join(get_ARGUS_dir(), 'Models')

def preload_itk_argus():
    # avoids "ITK not compiled with TBB" errors
    # this just does an instantiation prior to actuallly using it
    T = itk.Image[itk.F,2]
    itk.itkARGUS.ResampleImageUsingMapFilter[T,T].New()

preload_itk_argus()

# load argus stuff after ITK
sys.path.append(get_ARGUS_dir())
from ARGUS_LinearAR import *

class ArgusWorker:
    def __init__(self, sock, log):
        self.sock = sock
        self.log = log
        self.linearAR = ARGUS_LinearAR(model_dir=get_model_dir(), device_name='cpu')

    def run(self):
        msg = self.sock.recv()
        if msg.type != Message.Type.START:
            raise WorkerError('did not see start msg')
        
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            raise WorkerError('failed to parse start frame')
        
        video_file = data['video_file']
        debug = data.get('debug', False)

        try:
            if not path.exists(video_file):
                raise Exception(f'File {video_file} is not accessible!')

            stats = Stats()
            inf_result = self.linearAR.predict(video_file, stats=stats)
        except Exception as e:
            self.log.exception(e)
            error_msg = Message(Message.Type.ERROR, json.dumps(str(e)).encode('ascii'))
            self.sock.send(error_msg)
            return

        result = dict(
            sliding=inf_result['decision'] == 'Sliding',
            not_sliding_count=inf_result['not_sliding_count'],
            sliding_count=inf_result['sliding_count'],
            stats=stats.todict(),
        )
        result_msg = Message(Message.Type.RESULT, json.dumps(result).encode('ascii'))
        self.sock.send(result_msg)

        if debug:
            # save intermediate results
            save_path = f'{path.splitext(video_file)[0]}-debug-output'
            os.makedirs(save_path, exist_ok=True)

            itk.imwrite(itk.GetImageFromArray(inf_result['arnet_input_tensor'][0,0,:,:,:]),
                path.join(save_path, 'ARUNet_preprocessed_input.mha'))

            itk.imwrite(itk.GetImageFromArray(inf_result['arnet_output']),
                path.join(save_path, "ARUNet_output.mha"))

            itk.imwrite(itk.GetImageFromArray(inf_result['roinet_input_roi'].astype(np.float32)),
                path.join(save_path, "ROINet_input_roi.mha"))

            itk.imwrite(itk.GetImageFromArray(inf_result['roinet_input_tensor'][0,:,:,:]),
                path.join(save_path, "ROINet_preprocessed_input.mha"))

            itk.imwrite( itk.GetImageFromArray(inf_result['class_array']),
                path.join(save_path, "ARGUS_output.mha"))