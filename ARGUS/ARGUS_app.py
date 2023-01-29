#!/usr/bin/env python
# coding: utf-8

import sys

import itk
itk.force_load()

import monai

from ARGUS_app_ai import ARGUS_app_ai


if __name__ == "__main__":
    task = None
    source = None
    device_num = None
    if len(sys.argv) > 1:
        for arg in sys.argv[1:-1]:
            if arg in ARGUS_app_ai.tasks:
                task = arg
            elif arg in ARGUS_app_ai.sources:
                source = arg
            elif len(arg) == 1 and arg.isdigit():
                device_num = int(arg)
            else:
                print(f"ERROR: Option {arg} undefined.")
                quit()
        filename = sys.argv[-1]
    else:
        print("Usage: ARGUS_app [ultrasound_source] [task] <filename>")
        print("   ultrasound_source:")
        print("     ", ARGUS_app_ai.sources)
        print("   task:")
        print("     ", ARGUS_app_ai.tasks)
        #filename = "Data/ptx.mp4"
        #filename = "Data/pnb.mp4"
        #filename = "Data/onsd.mp4"
        filename = "Data/ett.mp4"
        print("")
        print(f"Running demo using {filename}")
        print("")

app_ai = ARGUS_app_ai(device_num=device_num, source=source)

result = app_ai.predict(filename, task=task)

if result != None:
    if result["decision"] == 0:
        print("Negative")
    else:
        print("Positive")
