require(devtools)
load_all()

X = matrix(rnorm(10000),1000,10); y = rnorm(1000)
cudaLassoPath(X,y,1:5)
