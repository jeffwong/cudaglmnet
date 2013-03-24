require(cudaglmnet)
require(glmnet)

n = 100; p = 10
X = matrix(rnorm(n*p),n,p); B.true = rnorm(p)
y = X %*% B.true + rnorm(n)

system.time(coef(cudaglmnet(X,y,lambda=100, maxIt = 500, standardize.x=F)))
system.time(coef(glmnet(X,y,lambda=1)))
