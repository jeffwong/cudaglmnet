dyn.load("src/GPULassoPath.so")
X = matrix(rnorm(1000),100,10); 
B.true = rnorm(10)
y = X %*% B.true; lambda = 0

type = 0
B = matrix(0, ncol(X), length(lambda))
maxIt = 5; threshold = 1e-6; gamma = 0.9; step_size = 10; reset = 30
  
  n <- nrow(X)
  p <- ncol(X)

  fit <- .C("activePathSol", X = as.single(X), y = as.single(y), n = as.integer(n), p = as.integer(p),
            lambda = as.single(lambda), num_lambda = as.integer(length(lambda)),
            type = as.integer(type), beta = as.single(B), maxIt = as.integer(maxIt),
            thresh = as.single(threshold), gamma = as.single(gamma), t = as.single(step_size),
            reset = as.integer(reset))


