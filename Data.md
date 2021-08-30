# Data

Standard directory structure expected to run the Experiments and to run
the tests within the ARGUS final project.

The Data subdir should be mounted from the data directory at Kitware.
The data cannot be shared beyond the participants on this project.

## Linux
ln -s ...

## Windows
In a CMD windows with Adminstrator privileges, replace <share> with
local share network name and C:\src with the path to your repo:
mklink /D C:\src\AnatomicRecon-POCUS-AI\Data \\<share>\other\projects\DARPA_POCUS_AI\Data
