[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.17025518-blue)](https://doi.org/10.5281/zenodo.17025518)

## Introduction

Hello everybody. Thanks for showing interest in this repository.

Feel free to download your version of EMerge and start playing around with it!
If you have suggestions/changes/questions either use the Github issue system or join the Discord using the following link:

**[Discord Invitation](https://discord.gg/VMftDCZcNz)**

## How to install

You can now install the basic version of emerge from PyPi!
```
pip install emerge
```
On MacOS and Linux you can install it with the very fast UMFPACK through scikit-umfpack

```
brew install cmake swig suite-sparse #MacOS
sudo apt-get install libsuitesparse-dev #Linux
```
Then on MacOS do:
```
export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"
export CFLAGS="-I/opt/homebrew/include"
export LDFLAGS="-L/opt/homebrew/lib"
```
Finally:
```
pip install emerge[umfpack]
```

### Experimental

If you have a new NVidia card you can try the first test implementation of the cuDSS solver. The dependencies can be installed through:
```
pip install emerge[cudss]
```
The `scikit-umfpack` solver can be installed on Windows as well from binaries with conda. This is a bit more complicated and is described in the installation guide which can be downloaded from the official website: 

https://www.emerge-software.com/resources

## Compatibility

As far as I know, the library should work on all systems. PARDISO is not supported on ARM but the current SuperLU and UMFPACK solvers work on ARM as well. Both SuperLU and UMFPACK can run on multi-processing implementations as long as you do entry-point protection:
```
import emerge as em

def main():
    # setup simulation

    model.mw.run_sweep(True, ..., multi_processing=True)

if __name__ == "__main__":
    main()
```
Otherwise, the parallel solver will default to SuperLU which can be slower on larger problems with a very densely connected/compact matrix.

## Required libraries

To run this FEM library you need the following libraries

 - numpy
 - scipy
 - gmsh
 - loguru
 - numba
 - matplotlib (for the matplotlib base display)
 - pyvista (for the PyVista base display)
 - cloudpickle
 - mkl (x86 devices only)

Optional:
 - scikit-umfpack
 - cudss
 - ezdxf

## Resources / Manual

You can find the latest versions of the manual on: **https://www.emerge-software.com/resources/**
