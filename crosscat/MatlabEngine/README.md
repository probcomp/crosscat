# MatlabEngine

MatlabEngine provides minimal support for the functions required for `geweke_utils.py`

## Caveats

- Requires [mlabwarp](http://mlabwrap.sourceforge.net/)
- You have to add the MatlabEngine folder (the one with the .m files) to the matlab path. You can do this with `mlab.addpath("path_string")` (see geweke_utils.py)
- Supports only normal (continuous) data
- Input data, `T`, must not have any columns s.t. `std(T[:,col]) = 0`
- Slow. Like really slow. You've been warned. 
- **Note:** The math is slightly different (the parametrization of Normal-Gamma is a little different) and is converted.

## Installing mlabwrap

1. Before you ask: no, you can't use pip.
2. [Download the files](http://sourceforge.net/projects/mlabwrap/) and run setup.py.
3. It probably doesn't work yet. If you're on OSX you're going to need to set some environmental variables in your virtual environment's `activatee` file.

```bash
export DYLD_LIBRARY_PATH=<path to .virtualenvs>/crosscat/lib/dylib
export MLABRAW_CMD_STR="/Applications/MATLAB_R2013a.app/bin/matlab -nodesktop"
```
then in `crosscat/lib/dylib` I have softlinks to the following libraries in the MATLAB.app bin

    libeng.dylib     
    libmat.dylib      
    libmex.dylib      
    libmx.dylib       
    libut.dylib

and `/usr/lib/libSystem.B.dylib`.

It should work now.