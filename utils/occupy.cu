#include <cuda.h>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>

// nvcc ./occupy.cu -o run

__device__ int get_global_index(void)
{
	return blockIdx.x * blockDim.x + threadIdx.x;
}

__global__ void kernel(void)
{
	while (1)
		;
}

int main(int argc, char **argv)
{
	int block_size = 128;
	int grid_size = 1;
	int gpu_num;

	// unsigned long int bytes = 10e9; // size of memory to occupy
	unsigned long int avail;
	unsigned long int total;
	float *data;

	cudaGetDeviceCount(&gpu_num);
	if (argc > 1)
	{
		for (int i = 1; i < argc; i++)
		{
            printf("occupying %d\n", atoi(argv[i]));
			cudaSetDevice(atoi(argv[i]));
			cudaMemGetInfo(&avail, &total);
			// cudaMalloc((void **)&data, avail - (9 - std::rand() % 5) * 1024 * 1024 * 1024);
            cudaMalloc((void **)&data, avail - 1024 * 1024 * 1024);
			kernel<<<grid_size, block_size>>>();
		}
	}
	else
	{
		for (int i = 0; i < gpu_num; i++)
		{
			cudaSetDevice(i);
			cudaMemGetInfo(&avail, &total);
			//cudaMalloc((void **)&data, avail - (9 - std::rand() % 5) * 1024 * 1024 * 1024);
            cudaMalloc((void **)&data, avail - 1024 * 1024 * 1024);
			kernel<<<grid_size, block_size>>>();
		}
	}

	cudaDeviceSynchronize();
    
	while (1)
		;
	return 0;
}