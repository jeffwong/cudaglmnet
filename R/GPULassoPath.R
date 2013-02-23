#' Lasso on a GPU
#'
#' Entry point to CUDA implementation of lasso
#' @param X design matrix X
#' @param y response vector y
#' @param B initial value for beta matrix for varying lambda penalty
#' @param lambda l1 penalties
#' @param standardize.x logical.  If true standardize the design matrix
#' @param standardize.y logical.  If true standardize the response vector
#' @param step_size step size for gradient descent
#' @param threshold convergence threshold
#' @param maxit maximum iterations
#' @useDynLib GPULassoPath
#' @export
cudaLassoPath <- function(X, y, lambda,
                          family = "gaussian",
                          B = matrix(0, ncol(X), length(lambda)),
                          standardize.x = T, standardize.y = T,
                          maxIt = 1e3, 
                          threshold = 1e-6, gamma = 1, step_size = 0.5, reset = 5) {
  
  n <- nrow(X)
  p <- ncol(X)

  if (standardize.x) {
    X = scale(X, center=F)
    X.sd = attr(X, "scaled:scale")
  } else X.sd = rep(1, ncol(X))

  if (standardize.y) {
    y = scale(y, center=F)
    y.sd = attr(y, "scaled:scale")
  } else y.sd = 1

  intercept = mean(y)
  y = y - intercept

  type = switch(family,
                "gaussian" = 0
               )

  fit <- .C("activePathSol",
            X = as.single(X), y = as.single(y), n = as.integer(n), p = as.integer(p),
            lambda = as.single(lambda), num_lambda = as.integer(length(lambda)),
            type = as.integer(type), beta = as.single(B), maxIt = as.integer(maxIt),
            thresh = as.single(threshold), gamma = as.single(gamma), t = as.single(step_size),
            reset = as.integer(reset),
            package = "RGPULasso")
  fit$X.sd = X.sd
  fit$intercept = intercept
  fit$y.sd = y.sd
  structure(fit, class="RGPULasso.cudalasso")

}

#' Lasso Coefficients
#'
#' @param cudalasso fit from cudaLassoPath
#' @param ... other params, not used
#' @method coef RGPULasso.cudalasso
#' @S3method coef RGPULasso.cudalasso
coef.RGPULasso.cudalasso = function(cudalasso, ...) {
  n = cudalasso$n
  p = cudalasso$p
  b = matrix(cudalasso$beta, n, p)

  #scale back
  fit$beta = diag(X.sd) %*% fit$beta
  fit$beta = rbind(intercept * y.sd, fit$beta)
}

setClass("RGPULasso.cudalasso", representation = "list", S3methods = T)
