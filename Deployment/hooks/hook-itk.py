from os import path
from PyInstaller.utils.hooks import get_package_paths

_, pkgpath = get_package_paths('itk')

datas = [
    (path.join(pkgpath, 'Configuration'), path.join('itk', 'Configuration')),
]