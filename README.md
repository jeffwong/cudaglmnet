cuLasso
=======

This package requires [NVIDIA's nvcc](https://developer.nvidia.com/cuda-downloads) > 4.0 to run.
A GPU with [CUDA Compute Capability](https://developer.nvidia.com/cuda-gpus) >= 1.3 is also required.

##Compiling

Use [autoconf](http://www.gnu.org/software/autoconf/) by changing into the root directory of this package and
issuing `autoconf` to compile configure.ac into the configure script, and to also autogenerate the Makefile.

`main.cu` has its own entry point which can be executed without R.  This is extremely helpful for debugging.
It is not necessary to run this in order to install the R package

```
cd src
make exec
```

##Installation

If CUDA > 4.0 is installed, then this package can be installed as a normal source package.

First, compile the package as above.  Then inside the R console issue

```
install.packages("path to source package", repos = NULL, type = "source")
```

##Debugging

The entry point in `main.cu` should be used for debugging.  This allows you to use cuda-gdb and cuda-memcheck.

```
cd src/
make exec
cuda-gdb main
```

```
cd src
make exec
cuda-memcheck main
```
