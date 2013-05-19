require(cudaglmnet)
require(glmnet)

#OLS

n = 1000; p = 1000; lambda=1:5
X = matrix(rnorm(n*p),n,p); B.true = rnorm(p)
y = X %*% B.true + rnorm(n,0,10)

system.time(cudacoefs <- coef(cudaglmnet(X,y,lambda=lambda, maxIt = 1500, standardize.x=T)))
system.time(glmnetcoefs <- coef(glmnet(X,y,lambda=lambda)))

cudacoefs
glmnetcoefs

#Binomial

n = 1000; p = 1000; lambda=seq(from=0,to=1,length.out=10)
X = matrix(rnorm(n*p),n,p); B.true = rnorm(p)
y = X %*% B.true + rnorm(n,0,10)
y2 = sapply(y, function(i) if (i > 0) 1 else 0)

system.time(cudacoefs <- coef(cudaglmnet(X,y2,lambda=lambda, maxIt = 1500, standardize.x=T, family="logit")))
system.time(glmnetcoefs <- coef(glmnet(X,y2,lambda=lambda, family="binomial")))

cudacoefs
glmnetcoefs

#Cox

n <- 500
p = 100
s <- 5

beta <- c(rep(1,s),rep(0,p-s))
X <- matrix(rnorm(n*p), ncol = p)
eta <- X %*% beta
eta1 <- rnorm(n)
start.time <- exp(eta1)
end.time <- start.time + exp(eta)
status <- rbinom(n,1,0.5)

lambda <- rep(100:1)/500

y <- list(start.time = start.time, end.time = end.time, status = status)

fit2 <- cudaglmnet(X = X, y = y, lambda = lambda, family = "cox")
coef(fit2)[,50]
