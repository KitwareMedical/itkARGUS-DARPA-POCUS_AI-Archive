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

pyinstaller argus.py from within the conda env