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
#' @export
cudaLassoPath <- function(X, y, B = matrix(0, ncol(X), length(lambda)),
                          lambda, standardize.x = T, standardize.y = T,
                          step_size= 0.1, threshold = 1e-6,
                          maxit = 1e3) {
  
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

  fit <- .C("activePathSol", X = as.single(X), y = as.single(y), n = as.integer(n),
            p = as.integer(p), maxIt = as.integer(maxit), thresh = as.single(threshold),
            step_size= as.single(step_size), lambda = as.single(lambda),
            beta = as.single(B), num_lambda = as.integer(length(lambda)))
  fit$beta <- matrix(fit$beta, nrow = p, byrow = F)

  #scale back
  fit$beta = diag(X.sd) %*% fit$beta
  fit$beta = rbind(intercept * y.sd, fit$beta)

  return (fit)
}
