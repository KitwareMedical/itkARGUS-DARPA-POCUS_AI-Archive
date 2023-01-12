#!/usr/bin/env python
# coding: utf-8

import sys

from ARGUS_app_ai import ARGUS_app_ai

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Default to using a Do_Not_Use case, since only CPU speed is being
        #   assessed, not network performance.
        filename = "Data/ptx.mp4"
    else:
        filename = sys.argv[1]
        print(filename)

app_ai = ARGUS_app_ai()
result = app_ai.predict(filename)
print(result["decision"])
