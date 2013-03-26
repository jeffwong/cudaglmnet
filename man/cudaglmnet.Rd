\docType{package}
\name{cudaglmnet}
\alias{cudaglmnet}
\alias{cudaglmnet-package}
\title{cudaglmnet}
\usage{
  cudaglmnet(X, y, lambda, family = "gaussian",
    B = matrix(0, ncol(X), length(lambda)),
    standardize.x = T, maxIt = 100, threshold = 1e-06,
    gamma = 0.5, step_size = 5, reset = 30)
}
\arguments{
  \item{X}{design matrix X}

  \item{y}{response vector y}

  \item{lambda}{l1 penalties}

  \item{B}{initial value for beta matrix for varying lambda
  penalty}

  \item{standardize.x}{logical.  If true standardize the
  design matrix}

  \item{maxit}{maximum iterations}

  \item{threshold}{convergence threshold}

  \item{gamma}{learning rate}

  \item{step_size}{step size for gradient descent}

  \item{reset}{}
}
\description{
  cudaglmnet

  Entry point to CUDA implementation of lasso
}

