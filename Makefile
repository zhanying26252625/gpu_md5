CUDA_INSTALL_PATH = /usr/local/cuda

INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include -I$(HOME)/NVIDIA_CUDA_SDK/common/inc -I$(CUDA_INSTALL_PATH)/samples/common/inc

LIBS = -L$(CUDA_INSTALL_PATH)/lib -L$(HOME)/NVIDIA_CUDA_SDK/lib

LIB64 = -L$(CUDA_INSTALL_PATH)/lib64

CC = gcc

NVCC = nvcc

CFLAGS            = -o3 -g

LDFLAGS           = -lrt -lm $(LIBS) $(LIB64) -lcudart -lstdc++ 

objects = main.o utility.o gpu_md5.o deviceQuery.o gpu_md5_gpu.cu_o 

 headers = $(wildcard *.h)

target:  gpu_md5

%.cu_o : %.cu
	$(NVCC) -c $(INCLUDES) -o $@ $<

%.o: %.cpp $(headers)
	$(CC) -c $(CFLAGS) $(INCLUDES) -o $@ $<

gpu_md5: $(objects)
	$(CC) -o $@ $^ $(LDFLAGS) $(INCLUDES) $(LIBS) $(LIB64)

clean:
	rm -f gpu_md5 *.o *.cu_o
