# -*- mode: python ; coding: utf-8 -*-

from glob import glob, iglob
from os import path
from PyInstaller import compat
from PyInstaller.utils.hooks import collect_submodules

def gather_mkl_dlls():
    dlls = []
    pattern = path.join(compat.base_prefix, '**', '*.dll')
    for filename in iglob(pattern, recursive=True):
        basename = path.basename(filename)
        if basename.startswith('mkl_'):
            dlls.append((filename, '.'))
    return dlls

def gather_ffmpeg_exes():
    exes = []
    pattern = path.join(compat.base_prefix, '**', '*.exe')
    for filename in iglob(pattern, recursive=True):
        basename = path.basename(filename)
        if basename == 'ffprobe.exe':
            exes.append((filename, '.'))
    return exes

block_cipher = None

binaries = []
# for numpy
binaries += gather_mkl_dlls()
# for ffmpeg-python
binaries += gather_ffmpeg_exes()

hiddenimports = []
hiddenimports += collect_submodules('av')

# server info
srv_a = Analysis(['server.py'],
             pathex=['C:\\Users\\Forrest\\AnatomicRecon-POCUS-AI\\Deployment'],
             binaries=binaries,
             datas=[],
             hiddenimports=hiddenimports,
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
srv_pyz = PYZ(srv_a.pure, srv_a.zipped_data,
             cipher=block_cipher)
srv_exe = EXE(srv_pyz,
          srv_a.scripts,
          [],
          exclude_binaries=True,
          name='argus-server',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )

# cli info
cli_a = Analysis(['cli.py'],
             pathex=['C:\\Users\\Forrest\\AnatomicRecon-POCUS-AI\\Deployment'],
             binaries=binaries,
             datas=[],
             hiddenimports=hiddenimports,
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
cli_pyz = PYZ(cli_a.pure, cli_a.zipped_data,
             cipher=block_cipher)
cli_exe = EXE(cli_pyz,
          cli_a.scripts,
          [],
          exclude_binaries=True,
          name='argus-cli',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True )

coll = COLLECT(srv_exe,
               srv_a.binaries,
               srv_a.zipfiles,
               srv_a.datas,
               cli_exe,
               cli_a.binaries,
               cli_a.zipfiles,
               cli_a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='argus')
