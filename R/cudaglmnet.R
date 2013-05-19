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
#' @param index index for grouping
#' @param penalty type of penalty "lasso" or "group"
#' @param alpha mixture for group penalty vs lasso penalty
#' @useDynLib cudaglmnet
#' @export
cudaglmnet <- function(X, y, lambda,
                       family = "gaussian",
                       B = matrix(0, ncol(X), length(lambda)),
                       standardize.x = T,
                       maxIt = 100, 
                       threshold = 1e-6, gamma = 0.5, step_size = 5, reset = 30, index = rep(1,ncol(X)), penalty = "lasso", alpha = 1, backtracking = TRUE) {
  
  n <- nrow(X)
  p <- ncol(X)

  if(penalty == "group"){

    ## This orders the groups by index
    
    ord <- order(index)
    index.ord <- index[ord]
    X <- X[,ord]
    unOrd <- match(ord, 1:length(ord))

    num.groups <- length(unique(index))
    begins <- rep(0, num.groups)
    ends <- rep(0, num.groups)

    ## This finds the beginning and ending index for each group (these are used in the C code)

    junk <- .C("set_inds",
               index = as.integer(index.ord), p = as.integer(p),
               begins = as.integer(begins), ends = as.integer(ends),
               package = "cudaglmnet")

    begins = junk$begins
    ends = junk$ends
  }

  if(penalty == "lasso"){
    index = 1
    num.groups = 1
    begins = 0
    ends = 0
  }
  
 
  type = switch(family,
                "gaussian" = 0,
                "logit" = 1,
                "cox" = 2
               )

  type_pen = switch(penalty,
                    "lasso" = 0,
                    "group" = 2 ## change to 1
                   ) 

  
  if(type == 0){
    intercept = mean(y)
    y = y-intercept
    setup.stuff <- setup.non.cox()
  }
  if(type == 1){
    intercept = 0
    setup.stuff <- setup.non.cox()
  }
  if(type == 2){ ##start.time, end.time, status

    intercept = 0

    ## Set up ordered times
    foo <- remove.extras(y$status, y$start.time, y$end.time)
    if(length(foo$ind.r) > 0){
      X <- X[-foo$ind.r,]
      n <- nrow(X)
      y$status <- y$status[-foo$ind.r]
      y$start.time <- y$start.time[-foo$ind.r]
      y$end.time <- y$end.time[-foo$ind.r]
    }
      
    setup.stuff <- setup.cox(y$status, y$start.time, y$end.time, foo$death.times)
    y = 1
  }

   if (standardize.x) {
    X = scale(X, center=F)
    X.sd = attr(X, "scaled:scale")
  } else X.sd = rep(1, p)
  

  fit <- .C("activePathSol",
            X = as.single(X), y = as.single(y), n = as.integer(n), p = as.integer(p),
            lambda = as.single(lambda*n), num_lambda = as.integer(length(lambda)),
            type = as.integer(type), beta = as.single(B), maxIt = as.integer(maxIt),
            thresh = as.single(threshold), gamma = as.single(gamma), t = as.single(step_size),
            reset = as.integer(reset), backtracking = as.integer(backtracking),
            type_pen = as.integer(type_pen),


            ### Group Lasso/Sparse Group Lasso Stuff
            alpha = as.single(alpha), num_groups = as.integer(num.groups),
            begins = as.integer(begins), ends = as.integer(ends),
            
            ### Cox Model Stuff
            s_person_ind = as.integer(setup.stuff$s.person.ind),
            e_person_ind = as.integer(setup.stuff$e.person.ind),
            status = as.integer(setup.stuff$status),
            s_risk_set = as.integer(setup.stuff$s.risk.set),
            e_risk_set = as.integer(setup.stuff$e.risk.set),
            nSet = as.integer(setup.stuff$nSet),

            package = "cudaglmnet")
  fit$X.sd = X.sd
  fit$intercept = intercept

  if(penalty == "group"){
    fit$X <- X[,unOrd]
    fit$X.sd <- fit$X.sd[unOrd]
    fit$beta = matrix(fit$beta, p, length(fit$lambda))
    fit$beta <- fit$beta[unOrd,]  
  }

  
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











remove.extras <- function(status, start.time, end.time){

  ### This finds the indices to remove
  ### (people who start after the last death or end before the first)
  
  death.times <- sort(unique(end.time[status == 1]))

  first.time <- death.times[1]
  last.time <- death.times[length(death.times)]

  ind.r <- union(which(end.time < first.time), which(start.time > last.time))
  return(list(ind.r = ind.r, death.times = death.times))
}



setup.cox <- function(status, start.time, end.time, death.times){

  n <- length(start.time)
  
  last.time <- death.times[length(death.times)]
  end.time[which(end.time > last.time)] <- last.time

  ord.s <- order(start.time)
  ord.e <- order(end.time, status)

  os.time <- start.time[ord.s]
  os.status <- status[ord.s]

  oe.time <- end.time[ord.e]
  oe.status <- status[ord.e]

  s.risk.set <- rep(0, n)
  e.risk.set <- rep(0, n)

  ind.t <- 1; ind.r <- 1
  while(ind.t <= n){
    if(os.time[ind.t] <= death.times[ind.r]){
      s.risk.set[ind.t] <- ind.r
      ind.t <- ind.t+1
    }
    else{
      ind.r <- ind.r + 1
    }
  }

  ind.t <- n; ind.r <- length(death.times)
  while(ind.t >= 1){
    if(oe.time[ind.t] >= death.times[ind.r]){
      e.risk.set[ind.t] <- ind.r
      ind.t <- ind.t - 1
    }
    else{
      ind.r <- ind.r - 1
    }
  }
return(list(s.person.ind = ord.s - 1, e.person.ind = ord.e - 1,status = status, s.risk.set = s.risk.set - 1, e.risk.set = e.risk.set - 1, nSet = length(death.times)))
}

setup.non.cox <- function(){
  return(list(s.person.ind = 0, e.person.ind = 0, status = 0, s.risk.set = 0, e.risk.set = 0))

}
