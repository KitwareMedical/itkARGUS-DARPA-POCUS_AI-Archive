#!/usr/bin/env python
# coding: utf-8

import sys

import itk
itk.force_load()

import monai

from ARGUS_app_ai import ARGUS_app_ai

if __name__ == "__main__":
    taskid = None
    if len(sys.argv) == 1:
        # Default to using a Do_Not_Use case, since only CPU speed is being
        #   assessed, not network performance.
        #filename = "Data/ptx.mp4"
        #filename = "Data/pnb.mp4"
        filename = "Data/onsd.mp4"
        #filename = "Data/ett.mp4"
        print("Usage: ARGUS_app [taskid] <filename>")
        print("   taskid:")
        print("       0 = PTX")
        print("       1 = PNB")
        print("       2 = ONSD")
        print("       3 = ETT")
    elif len(sys.argv) == 2:
        filename = sys.argv[1]
        print(f"Auto processing: {filename}")
    elif len(sys.argv) == 3:
        taskId = int(sys.argv[1])
        filename = sys.argv[2]
        print(f"Processing task={taskId}: {filename}")
    else:
        print("Usage: ARGUS_app [taskid] <filename")
        print("   taskid:")
        print("       0 = PTX")
        print("       1 = PNB")
        print("       2 = ONSD")
        print("       3 = ETT")

app_ai = ARGUS_app_ai()
result = app_ai.predict(filename, taskid=taskid)
print(result["decision"])
