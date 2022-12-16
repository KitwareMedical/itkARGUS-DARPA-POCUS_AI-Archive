#!/usr/bin/env python
# coding: utf-8

import os
import sys
import glob

assert len(sys.argv) == 3, 'usage: <searchdir> <outputdir>'

import itk
import numpy as np

import site
site.addsitedir("../ARGUS")

from ARGUS_IO import ARGUS_load_video

def main():
    for name in glob.iglob(f'{sys.argv[1]}/**/*.m??', recursive=True):
        dst = path.join(sys.argv[2].strip('/'), path.basename(name)[:-4]+".mha")
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        arr = ARGUS_load_video(name)
        img = itk.GetImageFromArray(arr).astype(np.float32)
        itk.imwrite(img, dst, compression=True)
        print(name, '>', dst)

main()
