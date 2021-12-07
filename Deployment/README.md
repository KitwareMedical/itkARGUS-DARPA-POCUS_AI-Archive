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

1. `pyinstaller argus.spec` from within the conda env, and verify that argus.exe works without an env.
2. Download NSSM and put the 32-bit `nssm.exe` file in this same directory.
3. Open the inno setup app and compile the final installer.

## running

For dev, you can run `server.py` or `argus-server.exe` for the server.

For prod, just run `argus-cli.bat` from the command prompt.
