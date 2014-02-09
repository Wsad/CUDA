#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <cuda_runtime.h>


void bitonicCmp(int *a, int *b){
	if (b > a){
		int t = *a;
		*a = *b;
		*b = t;
	}
}

/* 	Use array where numElements = 2^n (some power of 2)
*	Threads per block should be numElements/2
*	Not Working for sizes > 2048 (mayble limit per thread block)
*/
__global__ void bitonicSort(int *arr, int numElements){
	int id = threadIdx.x;
	int i = 2*id;
	int j = i + 1;
	if (arr[j] > arr[i]){
		int t = arr[i];
		arr[i] = arr[j];
		arr[j] = t;
	}
	syncthreads();
	
	for(int size = 4; size < numElements+1; size <<= 1){
		int step = id/(size/2);
		i = size*step + id - step*(size/2);
		j = i + size - 1 - 2*(id - step*(size/2));
		
		if (arr[j] > arr[i]){
			int t = arr[i];
			arr[i] = arr[j];
			arr[j] = t;
		}
		syncthreads();
		for(int s = size/2; s > 2; s >>= 1){
			step = id/(s/2);
			i = s*step + id - step*(s/2);
			j = i + s/2;
			if (arr[j] > arr[i]){
				int t = arr[i];
				arr[i] = arr[j];
				arr[j] = t;
			}			
			syncthreads();
		}
		i = 2*id;
		j = i + 1;
		if (arr[j] > arr[i]){
			int t = arr[i];
			arr[i] = arr[j];
			arr[j] = t;
		}
		syncthreads();
	}
}

int main(int argc, char *argv[]){
	int size;
	if (argc > 1)
		size = atoi(argv[1]);
	else
		exit(0);
	
	int *a = (int*)malloc(sizeof(int)*size);
	srand(NULL);
	for(int i =0; i < size; i++){
		a[i] = rand()%50;
	}
	
	int *d_a;
	cudaMalloc(&d_a, sizeof(int)*size);
	cudaMemcpy(d_a, a, sizeof(int)*size, cudaMemcpyHostToDevice);
	
	int threadsPerBlock = (size)/2;
	checkCudaErrors(bitonicSort<<< 1 , threadsPerBlock >>>(d_a, size));
	cudaDeviceSynchronize();
	
	cudaMemcpy(a, d_a, sizeof(int)*size, cudaMemcpyDeviceToHost);
	int last = a[0];
	for(int i =0; i < size; i++){
		if (a[i] > last){
			printf("Array is not sorted.\n");
			//printf("%d > %d\n", a[i], last);
			//last = a[i];
			exit(0);
		}else
			last = a[i];
	}
	printf("Array is sorted.\n");
	
	free(a);
	cudaFree(d_a);
	cudaDeviceReset();
	

}