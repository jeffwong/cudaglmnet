\name{cudaLassoPath}
\alias{cudaLassoPath}
\title{Lasso on a GPU}
\usage{
  cudaLassoPath(X, y,
    B = matrix(0, ncol(X), length(lambda)), lambda,
    standardize.x = T, standardize.y = T, step_size = 0.1,
    threshold = 1e-06, maxit = 1000)
}
\arguments{
  \item{X}{design matrix X}

  \item{y}{response vector y}

  \item{B}{initial value for beta matrix for varying lambda
  penalty}

  \item{lambda}{l1 penalties}

  \item{standardize.x}{logical.  If true standardize the
  design matrix}

  \item{standardize.y}{logical.  If true standardize the
  response vector}

  \item{step_size}{step size for gradient descent}

  \item{threshold}{convergence threshold}

  \item{maxit}{maximum iterations}
}
\description{
  Entry point to CUDA implementation of lasso
}

