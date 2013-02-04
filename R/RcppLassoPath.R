#' Rcpp Lasso
#' 
#' Implementation of lasso using majorize minimization in Rcpp
#' @param X
#' @param y
#' @param B
#' @param stepsize
#' @param lambda
#' @param max.iters
#' @useDynLib RGPULasso
#' @export
RcppLassoPath = function(X, y, B = matrix(0,nrow(X),length(lambda)),
                         lambda, standardize.x = T, standardize.y = T,
                         stepsize = 1e-4, max.iters = 100) {
  if (standardize.x) {
    X.scale = scale(X)
  } else X.scale = X

  if (standardize.y) {
    y.scale = scale(y, center=F)
    intercept = mean(y.scale)
  } else {
    y.scale = y
    intercept = mean(y)
  }

  lasso.rcpp = RcppLasso(X.scale,y.scale,B,lambda,stepsize,max.iters)
  lasso.rcpp$coefficients = rbind(intercept, lasso.rcpp$coefficients)
  lasso.rcpp
}
