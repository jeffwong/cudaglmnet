EXT := cu 

OBJS := main.o 
TARGETS := GPULassoPath.so

#OBJS13 := @OBJS13@ 

CULA_OBJS := 
CULA_INCS := -I/usr/local/cula/include

#compiler/preprocessor options
INCS := -I. -I/usr/local/cuda/include -I/Library/Frameworks/R.framework/Resources/include -I/Library/Frameworks/R.framework/Resources/include/x86_64
PARAMS :=  -m64 -Xcompiler -fPIC

#linker options
LD_PARAMS := -m64 -Xlinker -rpath,/usr/local/cuda/lib,'-F/Library/Frameworks/R.framework/.. -framework R'
LIBS :=   -L/usr/local/cuda/lib -lcublas

NVCC := /usr/local/cuda/bin/nvcc #-gencode arch=compute_10,code=sm_10 -gencode arch=compute_11,code=sm_11

all: $(TARGETS) 

$(TARGETS): $(OBJS13) $(CULA_OBJS) $(OBJS)
	$(NVCC) -O2 -shared $(LD_PARAMS) $(LIBS) $(OBJS13) $(CULA_OBJS) $(OBJS) -o $@

$(OBJS): %.o: %.$(EXT)
	$(NVCC) -O2 -c $(INCS) $(PARAMS) $^ -o $@

$(OBJS13): %.o: %.$(EXT)
	$(NVCC) -O2 -c $(INCS) $(PARAMS) $^ -o $@

$(CULA_OBJS): %.o: %.c
	$(NVCC) -O2 -c $(INCS) $(CULA_INCS) $(PARAMS) $^ -o $@

exec:
	$(NVCC) -DTHRUST main.cu -o main $(LIBS)

clean:
	rm -rf *.o; rm -rf *.so
	rm main

.PHONY: all clean
