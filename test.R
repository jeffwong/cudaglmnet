require(cudaglmnet)
require(glmnet)

n = 1000; p = 10; lambda=1
X = matrix(rnorm(n*p),n,p); B.true = rnorm(p)
y = X %*% B.true + rnorm(n,0,10)

system.time(cudacoefs <- coef(cudaglmnet(X,y,lambda=lambda, maxIt = 1500, standardize.x=T)))
system.time(glmnetcoefs <- coef(glmnet(X,y,lambda=lambda)))

cudacoefs
glmnetcoefs

