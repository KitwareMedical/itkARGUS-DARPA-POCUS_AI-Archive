# Data

Standard directory structure expected to run the Experiments and to run
the tests within the ARGUS final project.

We recommend having both local copies and network mounted versions of 
the data readily available.

The network mounsted verions should be mounted from shared drives at
Kitware.  The data cannot be shared beyond the participants on this project.

The general structure expected is
PNB/Data_PNB/...<local copy>...
PNB/Data_PNB_Net/...<mounted copy>...

## Mounting using Linux
> ln -s ...

## Mounting using Windows
In a CMD windows with Adminstrator privileges, replace <share> with
local share network name and C:\src with the path to your repo:
> mklink /D C:\src\AnatomicRecon-POCUS-AI\PNB\Data_PNB_Net \\<share>\other\projects\DARPA_POCUS_AI\Data_PNB
