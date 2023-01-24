ARGUS-AI: Anatomic Reconstruction for Generalized UltraSound AI
Kitware, Inc. and Duke University


Installation
============
1. Download the “argus-installer-v2.0.beta01.exe” installer to our evaluation machine (the Microsoft Surface Book 3).

2. Double-click “argus-installer-v2.0.beta01.exe” and follow the installation prompts. The defaults should work, so click “Next” to go through the installation. The installer will install our “AI” service and update the system path to include our executable.



To Evaluate a File
==================

1. Open a windows command prompt (“cmd.exe”). You can accomplish this by pressing the “Windows + R” keys, and then typing “cmd.exe” and hitting Enter. This should open a new command prompt window located in “C:\Users\<YourUserName>”.

2. Type the following command, where “<path\to\video.mp4>” should be replaced with a valid path to a video mp4/mov file.

      > argus-cli -f <path\to\video.mp4>

After a moment, you should receive the following output, which indicates that the video was read and a sample inference was performed. (Note that for this initial deployment test, we are not yet running our actual inference engines.)

      File: <filename>
         Task: <PTX|PNB|ONSD|ETT>
         Prediction: <0|1>
            Confidence Measure 0: <value>
            Confidence Measure 1: <value>

Additionally a file will be created in the same folder as filename, with the suffix ".csv".



To Evaluate a Directory of Files   
================================

1. Open a windows command prompt (“cmd.exe”). You can accomplish this by pressing the “Windows + R” keys, and then typing “cmd.exe” and hitting Enter. This should open a new command prompt window located in “C:\Users\<YourUserName>”.

2. Type the following command, where “<path\to\directory>” should be replaced with a valid path to directory in which every video (*m??) file will be evaluated.

      > argus-cli -d <path\to\directory>

After a moment, the system will begin processing each video file in sequence.  The output displayed and the csv file generated for each video file will be the same as if they were processed individually



Specifying the Task
===================

The ARGUS-AI system uses an image classification neural network to automatically determine the anatomy being viewed and thereby select the task to be performed (i.e., PTX, PNB, ONSD or ETT analysis).

Optionally, you can override the automatic task identifier and specify the task to be performed using the -t option:

      > argus-cli -t <task_numeric_id> -[f|d] <file|directory>

The tasks have been assigned the following numeric ids:

         0 = PTX
         1 = PNB
         2 = ONSD
         3 = ETT


 
Errors
======

If an error occurs, the cli will do its best to tell you what happened. When reporting errors back to us, please include both the cli output and the contents of the log file located at “C:\Program Files (x86)\ARGUS\argus\server-log.log” in your report.
