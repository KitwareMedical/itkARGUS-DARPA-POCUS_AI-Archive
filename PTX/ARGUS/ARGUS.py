#!/usr/bin/env python
# coding: utf-8

import sys

from ARGUS_LinearAR import *

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Default to using a Do_Not_Use case, since only CPU speed is being
        #   assessed, not network performance.
        filename = "../Data/Final15/BAMC-PTXSliding/212s_image_128692595484031_CLEAN.mov"
    else:
        filename = sys.argv[1]
        print(filename)

argus = ARGUS_LinearAR()
result = argus.predict(filename)
print(result["decision"])
