

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include "c_utils.h"
#include "des.h"
#include "des_utils.h"
#include "bit_utils.h"
#include "des_consts.h"
#include "des_kernel.h"
#include "cuda_utils.h"


static void CheckCudaErrorAux(const char *, unsigned, const char *,
    cudaError_t);
#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)

/**
 * Check the return value of the CUDA runtime API call and exit
 * the application if the call has failed.
 */
static void CheckCudaErrorAux(const char *file, unsigned line,
    const char *statement, cudaError_t err) {
  if (err == cudaSuccess)
    return;
  std::cerr << statement << " returned " << cudaGetErrorString(err) << "("
      << err << ") at " << file << ":" << line << std::endl;
  exit(1);
}

__global__ void kernel(int* resultsDevice, int dim, uint64_t* hashesDevice) {
	int mI = threadIdx.y+blockIdx.y*blockDim.y;
	int yI = threadIdx.x+blockIdx.x*blockDim.x + 1940;
	int dI = threadIdx.z;
	uint64_t key = yI*10000+mI*100+dI;
	uint64_t encoded = 0;
	encoded = full_des_encode_block(key, key);
	for(int i=0;i<dim;i++){
		if (hashesDevice[i] == encoded){
			resultsDevice[i] = 1;
		}
	}
}


int main(void){
	#define dim 1000
	int resultsHost[dim];
	FILE * fp;
	char * line = NULL;
	size_t len = 0;
	ssize_t read;
	uint64_t hashesHost[dim];
	int k=0;
	fp = fopen("PswDb/db1000.txt", "r");
	while ((read = getline(&line, &len, fp)) != -1) {
		char* hash =(char*) malloc(sizeof(char)*9);
		for(int i = 0; i<9; i++){
		  hash[i]=line[i];
		}
		hash[8]= '\0'; //string termination
		hashesHost[k]=full_des_encode_block(atoi(hash),atoi(hash));
		k++;
	}
	fclose(fp);
	free(line);

	//GPU memory allocation
	uint64_t* hashesDevice;
	int* resultsDevice;

	CUDA_CHECK_RETURN( cudaMalloc((void **)&hashesDevice, dim * sizeof(uint64_t)) );

	CUDA_CHECK_RETURN( cudaMemcpy(hashesDevice, hashesHost, dim * sizeof(uint64_t), cudaMemcpyHostToDevice) );

	CUDA_CHECK_RETURN( cudaMalloc((void **) &resultsDevice, sizeof(int) * dim));

	dim3 dimGrid(8,5);
	dim3 dimBlock(10,3,32);//
	clock_t start = clock();
	kernel<<<dimGrid,dimBlock>>>(resultsDevice,dim,hashesDevice);
	// copy results from device memory to host
	cudaDeviceSynchronize();
	CUDA_CHECK_RETURN(
	  cudaMemcpy(resultsHost, resultsDevice, dim * sizeof(int),
		  cudaMemcpyDeviceToHost));
	clock_t end = clock();
	float seconds = (float) (end - start) / CLOCKS_PER_SEC;
	cudaFree(hashesDevice);
	cudaFree(resultsDevice);

	int count = 0;
	for(int i = 0; i < dim; i++){
		if(resultsHost[i] == 1){
			count++;
		}
	}
	printf("found hashes: %d\n", count);
	printf("time: %f",seconds);
	return 0;
}
