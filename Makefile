CXX = nvcc
CULDFLAGS=-L/usr/local/cuda/lib
CFLAGS = -O2  -I $(CUDA_ROOT)/include -I${MKLROOT}/include  -I /home/szhang41/lls/RewriteLLS
LFLAGS = -L $(CUDA_ROOT)/lib64 -lcusolver -lcublas -lcurand -lcudart -lcusparse  -lcuda  -Xlinker -rpath,${MKLROOT}/lib  -lmkl_rt -lpthread -lm -ldl -lineinfo
CC = gcc

lls: lls.cu
	$(CXX)  $(CFLAGS) $(LFLAGS) $^ -o $@

cgls: cgls.cu cgls.cuh
	nvcc -O3 -m64 -arch=sm_30 -o $@ -lcublas -lcusparse $(CULDFLAGS) $<