#' Lasso on a GPU
#'
#' Entry point to CUDA implementation of lasso
#' @param X design matrix X
#' @param y response vector y
#' @param lambda l1 penalties
#' @param B initial value for beta matrix for varying lambda penalty
#' @param standardize.x logical.  If true standardize the design matrix
#' @param maxit maximum iterations
#' @param threshold convergence threshold
#' @param gamma learning rate
#' @param step_size step size for gradient descent
#' @param reset
#' @useDynLib cudaglmnet
#' @export
cudaglmnet <- function(X, y, lambda,
                       family = "gaussian",
                       B = matrix(0, ncol(X), length(lambda)),
                       standardize.x = T,
                       maxIt = 100, 
                       threshold = 1e-6, gamma = 0.5, step_size = 5, reset = 30) {
  
  n <- nrow(X)
  p <- ncol(X)

  if (standardize.x) {
    X = scale(X, center=F)
    X.sd = attr(X, "scaled:scale")
  } else X.sd = rep(1, p)

  intercept = mean(y)
  y = y-intercept
  
  type = switch(family,
                "gaussian" = 0
               )

  fit <- .C("activePathSol",
            X = as.single(X), y = as.single(y), n = as.integer(n), p = as.integer(p),
            lambda = as.single(lambda*n), num_lambda = as.integer(length(lambda)),
            type = as.integer(type), beta = as.single(B), maxIt = as.integer(maxIt),
            thresh = as.single(threshold), gamma = as.single(gamma), t = as.single(step_size),
            reset = as.integer(reset),
            package = "cudaglmnet")
  fit$X.sd = X.sd
  fit$intercept = intercept
  structure(fit, class="cudaglmnet.cudalasso")
}

#' Lasso Coefficients
#'
#' @param cudalasso fit from cudaLassoPath
#' @param ... other params, not used
#' @method coef RGPULasso.cudalasso
#' @S3method coef cudaglmnet.cudalasso
coef.cudaglmnet.cudalasso = function(cudalasso, ...) {
  n = cudalasso$n
  p = cudalasso$p
  b = matrix(cudalasso$beta, p, length(cudalasso$lambda))

  #scale back
  apply(b, 2, function(j) { cudalasso$X.sd * j })
  b = rbind(cudalasso$intercept, b)

  return (b)
}

setClass("cudaglmnet.cudalasso", representation = "list", S3methods = T)
