cuLasso
=======

This package requires [NVIDIA's nvcc](https://developer.nvidia.com/cuda-downloads) > 4.0 to run

#Compiling

Use [autoconf](http://www.gnu.org/software/autoconf/) by changing into the root directory of this package and
issuing `autoconf` to compile configure.ac into the configure script, and to also autogenerate the Makefile.

#Installation

If CUDA > 4.0 is installed, then this package can be installed as a normal source package.

First, compile the package as above.  Then inside the R console issue

```
install.packages("path to source package", repos = NULL, type = "source")
```
