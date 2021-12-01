## Installation

```
conda create -n service
conda activate service
pip install pywin32
conda install -c conda-forge pyinstaller
# for argus
conda install -c conda-forge ffmpeg ffmpeg-python av numpy
```

## build

`pyinstaller argus.spec` from within the conda env, and verify that argus.exe works without an env.

```
argus.exe --server
argus.exe path/to/video.mp4
```