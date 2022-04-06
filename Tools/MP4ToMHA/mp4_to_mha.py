#!/usr/bin/env python
# coding: utf-8

import sys
import os
from os import path
import glob

assert len(sys.argv) == 3, 'usage: <searchdir> <outputdir>'

import cv2
import itk
import numpy as np


image_type = itk.Image[itk.UC,2]
image_type_out = itk.Image[itk.UC,3]

def mp4tolist(fname):
    imlist = []

    cap = cv2.VideoCapture(fname)
    if cap.isOpened():

        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            img_slice = itk.image_from_array(np.array(gray))
            imlist.append(img_slice)
        cap.release()
 
    return imlist

def dojoin(l):
    join = itk.JoinSeriesImageFilter[image_type, image_type_out].New()
    for i, a in enumerate(l):
        join.SetInput(i, a)
    join.Update()
    return join.GetOutput()


def main():
    for name in glob.iglob(f'{sys.argv[1]}/**/*.mp4', recursive=True):
        dst = path.join(sys.argv[2].strip('/'), path.basename(name)).replace('.mp4', '.mha')
        os.makedirs(path.dirname(dst), exist_ok=True)
        slice_list = mp4tolist(name)
        if len(slice_list) == 0:
            print('WARNING: File produced zero frames:', name)
            continue
        mp4image = dojoin(slice_list)
        itk.imwrite(mp4image, dst, compression=True)
        print(name, '>', dst)

main()
