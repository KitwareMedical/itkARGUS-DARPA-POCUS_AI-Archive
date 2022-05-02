itkARGUS
=================================

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :target: https://github.com/KitwareMedical/AnatomicRecon-POCUS-AI/blob/master/LICENSE
    :alt: License

Overview
--------

Contains C++ methods used by Kitware's Anatomic Reconstruction for Generalized UltraSound project.  These methods are implemented in C++ for speed, and then wrapped into python for use in the project.

Build Steps
-----------

Option 1: If you don't have a from-source build of ITK:
1. `git clone --depth=1 https://github.com/InsightSoftwareConsortium/ITKPythonPackage`
2. `./ITKPythonPackage/scripts/dockcross-manylinux-download-cache-and-build-module-wheels.sh`
  - If the command errors with "curl: failed to write body", then `chown -R` the `tools/` folder to be your user. Then, re-run the above command.
3. To rebuild, run `./ITKPythonPackage/scripts/dockcross-manylinux-build-module-wheels.sh`

Option 2: If oyu have a from-source build of ITK:
1. `git clone https://github.com/KitwareMedical/AnatomicRecon-POCUS-AI`
2. `mkdir itkARGUS-Release`
3. `cd itkARGUS-Release`
4. `cmake-gui ../AnatomicRecon-POCUS-AI`
  - Make certain to specify CMAKE_BUILD_TYPE=Release.
  - It should automatically find your ITK build directory
5. Compile the project.  It should then be wrapped for python and
  available via import itk.

Usage
-----

Sample usage can be found in `usage/resample.py`. It expects a US video in `video.mp4` and outputs a `output.nrrd` file. Be sure to install the packages found in requirements.txt.
