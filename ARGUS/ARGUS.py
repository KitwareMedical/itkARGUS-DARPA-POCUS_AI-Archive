#!/usr/bin/env python
# coding: utf-8

from ARGUS_LinearAR import *

# Default to using a Do_Not_Use case, since only CPU speed is being
#   assessed, not network performance.
#filename = "../Data/Final15/BAMC-PTXNoSliding/Do_Not_Use/219ns_image_1895283541879_clean.mov"
filename = "../Data/Final15/BAMC-PTXSliding/212s_image_128692595484031_CLEAN.mov"

# Preload everything
print("Preloading")
ARGUS_LinearAR(filename)

# Run for timing
print("Running")
with time_this("Total run time",True):
    ARGUS_LinearAR(filename,"cpu",False)

# Run for debugging
print("Component timing")
ARGUS_LinearAR(filename,"cpu",True)
