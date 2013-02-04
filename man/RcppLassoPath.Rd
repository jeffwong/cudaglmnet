\name{RcppLassoPath}
\alias{RcppLassoPath}
\title{Rcpp Lasso}
\usage{
  RcppLassoPath(X, y,
    B = matrix(0, nrow(X), length(lambda)), lambda,
    standardize.x = T, standardize.y = T, stepsize = 1e-04,
    max.iters = 100)
}
\arguments{
  \item{X}{}

  \item{y}{}

  \item{B}{}

  \item{stepsize}{}

  \item{lambda}{}

  \item{max.iters}{}
}
\description{
  Implementation of lasso using majorize minimization in
  Rcpp
}

