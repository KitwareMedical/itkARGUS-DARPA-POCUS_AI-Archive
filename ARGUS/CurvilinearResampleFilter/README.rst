CurvilinearResampleFilter
=================================

.. image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
    :target: https://github.com/KitwareMedical/AnatomicRecon-POCUS-AI/blob/master/LICENSE
    :alt: License

Overview
--------

Resamples curvilinear ultrasound images

Build Steps
-----------

1. `git clone --depth=1 https://github.com/InsightSoftwareConsortium/ITKPythonPackage`
2. `./ITKPythonPackage/scripts/dockcross-manylinux-download-cache-and-build-module-wheels.sh`
  - If the command errors with "curl: failed to write body", then `chown -R` the `tools/` folder to be your user. Then, re-run the above command.
3. To rebuild, run `./ITKPythonPackage/scripts/dockcross-manylinux-build-module-wheels.sh`

Usage
-----

Sample usage can be found in `usage/resample.py`. It expects a US video in `video.mp4` and outputs a `output.nrrd` file. Be sure to install the packages found in requirements.txt.
