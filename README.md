cudaglmnet
=======

glmnet is a machine learning library in R that estimates generalized linear models with L1 and L2 regularizations.
CUDA is a NVIDIA programming language allowing developers to make use of GPUs for general purpose programming.
GPUs have hundreds of cores, allowing them to run the complex mathematical algorithms needed for video processing in
animation, movies, games, etc.  This R library does the computations for glmnet on the GPU, allowing massive parallel
computations.

This package requires [NVIDIA's nvcc](https://developer.nvidia.com/cuda-downloads) > 4.0 to run

#Compiling and Installing

```
./configure
R CMD INSTALL ./
```

#Example
```
require(cudaglmnet)

n = 1000; p = 10; lambda=1
X = matrix(rnorm(n*p),n,p); B.true = rnorm(p)
y = X %*% B.true + rnorm(n,0,10)

system.time(cudacoefs <- coef(cudaglmnet(X,y,lambda=lambda, maxIt = 1500, standardize.x=T)))

cudacoefs

```
