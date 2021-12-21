## Installation

```
conda create -n service
conda activate service
pip install pywin32
# do not install from conda-forge, as we need a more recent version
pip install pyinstaller=4.7
# for argus
pip install monai itk
conda install -c conda-forge ffmpeg ffmpeg-python av numpy
```

Edit the run function in `worker.py` to run your own code.

## build

1. `pyinstaller argus.spec` from within the conda env, and verify that argus.exe works without an env.
2. Download NSSM and put the 32-bit `nssm.exe` file in this same directory.
3. Open the inno setup app and compile the final installer.

## running

For dev, you can run `server.py` for the server and `cli.py` for the cli.

For a prod installation, the server will be autostarted for you. All you need to run is `argus-cli` from the command prompt.