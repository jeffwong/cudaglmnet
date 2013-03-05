dyn.load("GPULassoPath.so")

.C("foo", x = as.single(1:5), size=as.integer(5))
