//#include "stdafx.h"
// Sequence Alignment -CUDA
// Alex Ringeri

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define numThreads 128

void testReduce();

void CalculateCost(int *d_matrix, int *d_trace, char *d_s1, char *d_s2, int s1, int s2);

void CopyToMatrix(int *dst, int *src, int cols, int rows);

void PrintMatrix(int *arr, char *s1, char *s2, int xDim, int yDim);

//Kernel initializes all elements of matrix to 'value'
__global__ void init_matrix(int *matrix, int value, int maxElements){
	if (blockDim.x * blockIdx.x + threadIdx.x < maxElements)
		matrix[ blockDim.x * blockIdx.x + threadIdx.x] = value;
}

//Finds max value of array and places its index into '*iOut'
//Credit to udacity - Lesson 3 - reduction
__global__ void maxReduce(int *table, int *maxOut, int *iOut){
	int id = threadIdx.x;
	extern __shared__ int s_table[];
	
	//copy from global to shared memory
	s_table[id] = table[id];
	syncthreads(); 
	
	//Do first reduction and store indices in 2nd half of array
	unsigned int i = blockDim.x/2;
	if (id < i){
		if (s_table[id+i] > s_table[id]){
			s_table[id] = s_table[id + i];
			s_table[id+i] = id + i;
		}
		else
			s_table[id+i] = id;
		syncthreads();
		/*if (id ==0){
			for (int j=0; j < blockDim.x; j++)
				printf("%d ", s_table[j]);
			printf("\n");
		}
		syncthreads();*/
	}
	
	int temp = i;
	for(i >>= 1; i > 0; i>>=1){
		if (id < i){
			if (s_table[id+i] > s_table[id]){
				s_table[id] = s_table[id + i];
				s_table[id+i] = s_table[id+i+temp];
			}
			else
				s_table[id+i] = s_table[id + temp];
		}
		temp = i;
		syncthreads();
		/*if (id ==0){
			for (int j=0; j < blockDim.x; j++)
				printf("%d ", s_table[j]);
			printf("\n");
		}
		syncthreads();*/
	}
	
	//place resulting index into out
	if (id == 0){
		maxOut[blockIdx.x] = s_table[id];
		iOut[blockIdx.x] = s_table[id+1];
		//printf("BlockId: %d\tMax: %d\tIndex: %d\n", blockIdx.x, s_table[id], s_table[id+1]);
	}
}

void testReduce(){
	int size;
	scanf("%d", &size);
	int *arr = (int*)malloc(sizeof(int)*size);
	srand(time(NULL));
	for (int i=0; i <size; i++)
		arr[i] = rand()%500;
	
	int *d_i, *d_arr, *d_max;
	 
	cudaMalloc((void**)&d_arr, sizeof(int)*size);
	cudaMemcpy(d_arr, arr, sizeof(int)*size, cudaMemcpyHostToDevice);
	
	int threadsPerBlock = 1024;
	int blocks = (size + threadsPerBlock - 1)/threadsPerBlock;
	cudaMalloc((void**)&d_i, sizeof(int)*blocks); 
	cudaMalloc((void**)&d_max, sizeof(int)*blocks);	
	

	maxReduce<<< blocks, threadsPerBlock, sizeof(int)*threadsPerBlock>>>(d_arr, d_max, d_i);
	cudaDeviceSynchronize();
	cudaMemcpy(intermediateIndex, d_i, sizeof(int)*blocks, cudaMemcpyDeviceToHost);
	int *intermediateIndex = (int*)malloc(sizeof(int)*blocks);
	
	
	threadsPerBlock = blocks;
	blocks = 1;
	int *d_maxVal; int *d_interI;
	cudaMalloc((void**)&d_maxVal, sizeof(int)); 
	cudaMalloc((void**)&d_interI, sizeof(int));
	maxReduce<<< blocks, threadsPerBlock, sizeof(int)*threadsPerBlock>>>(d_max, d_maxVal, d_interI);
	cudaDeviceSynchronize();
	
	int *i = (int*)malloc(sizeof(int));
	int *j = (int*)malloc(sizeof(int));
	cudaMemcpy(i, d_interI, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(j, d_maxVal, sizeof(int), cudaMemcpyDeviceToHost);

	printf("Kernel: max %d index %d\n", *j, intermediateIndex[*i]);
	int max = 0; int in = 0;
	for (int j=0; j <size; j++){
		if (arr[j] >= max){
			max = arr[j];
			in = j;
		}
	}
	printf("Host: max %d index %d\n", max, in);
	
	free(i); free(arr); free(j); free(intermediateIndex);
	cudaFree(d_arr); cudaFree(d_i); cudaFree(d_max); cudaFree(d_maxVal); cudaFree(d_interI);
	
}

__global__ void ComputeDiagonal(int i, int prevI, int lastI, int space, int *arr, int *trace, char *s1, char *s2, int s1off, int s2off){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < space){
		int left = arr[prevI + id];
		int up = arr[prevI + id + 1];
		int upLeft = arr[lastI + id];
		
		if(s1[s1off + id] == s2[s2off - id] )
			upLeft += 2;
		else
			upLeft -= 1;

		int cost, dir;
		if (up > left){
			cost = up - 1;
			dir = 1;
		}
		else{
			cost = left - 1;
			dir = -1;
		}
		if (upLeft > cost){
			cost = upLeft;
			dir = 0;
		}
		arr[i + id] = max(cost, 0);
		trace[i + id] = dir;	
	}
}

// Main routine: Executes on the host
int main(int argc, char *argv[]){
	char AGCT[] = "AGCT";
	int lenS1, lenS2;
	if (argc > 2){
		int args[] = { atoi(argv[1]), atoi(argv[2]) };
		if (args[0] > args[1]){
			lenS1 = args[0];
			lenS2 = args[1];
		}else {
			lenS1 = args[1];
			lenS2 = args[0];
		}
	}else {
		printf("Invalid Command Line Arguments --- Exiting Program");
		exit(0);
	}

	//Allocate strings on host
	char * string1 = (char*)malloc(sizeof(char)*lenS1);
	char * string2 = (char*)malloc(sizeof(char)*lenS2);

	//Initialize strings with random numbers
	srand(time(NULL));
	for(int i=0; i<lenS1 ;i++)
		string1[i] = AGCT[rand()%4];
	for(int i=0; i<lenS2 ;i++)
		string2[i] = AGCT[rand()%4];
	
	//Allocate strings on device
	cudaError_t error = cudaSuccess;
	char *d_string1, *d_string2;
	
	error = cudaMalloc((void**)&d_string1, sizeof(char)*lenS1);
	
	if (error != cudaSuccess) {
		printf("Error allocating s1 on device\n");
		exit(0);
	}
	
	error = cudaMalloc((void**)&d_string2, sizeof(char)*lenS2);
	
	if (error != cudaSuccess) {
		printf("Error allocating s2 on device\n");
		exit(0);
	}
	
	//Initialize sequence strings on device
	error = cudaMemcpy(d_string1, string1, sizeof(char)*lenS1, cudaMemcpyHostToDevice);
	
	if (error != cudaSuccess) {
		printf("Error copying s1 to device\n");
		exit(0);
	}
	
	error = cudaMemcpy(d_string2, string2, sizeof(char)*lenS2, cudaMemcpyHostToDevice);
	
	if (error != cudaSuccess) {
		printf("Error copying s2 to device\n");
		exit(0);
	}

	//Allocate score table on Device
	int entries = (lenS1+1)*(lenS2+1);
	int* d_matrix;
	error = cudaMalloc((void**)&d_matrix, sizeof(int)*entries);
	
	if (error != cudaSuccess) {
		printf("Error allocating d_matrix on device\n");
		exit(0);
	}

	int threadsPerBlock = 256;
	int blocksPerGrid = (entries + threadsPerBlock -1)/threadsPerBlock;
	
	//Initialize score table with 0
	init_matrix<<< blocksPerGrid, threadsPerBlock >>>(d_matrix, 0, entries);
	
	
	//Allocate trace table on Device
	int* d_trace;
	error = cudaMalloc((void**)&d_trace, sizeof(int)*entries);
	
	if (error != cudaSuccess) {
		printf("Error allocating d_matrix on device\n");
		exit(0);
	}

	//Initialize trace table with -2
	init_matrix<<< blocksPerGrid, threadsPerBlock >>>(d_trace, -2, entries);

	/* Do calculation on device:
	 *
	 */
	CalculateCost(d_matrix, d_trace, d_string1, d_string2, lenS1+1, lenS2+1);
	cudaDeviceSynchronize();
	
	error = cudaGetLastError();
	
	if (error != cudaSuccess) {
		printf("Error with kernel or d_matrix/d_trace allocation: %s\n", cudaGetErrorString(error));
		exit(0);
	}
	
	testReduce();
	/*int *posMax;
	error = cudaMalloc((void**)&posMax, 1*sizeof(int));
	maxReduce<<< 1, 1024, 2048*sizeof(int) >>>(d_matrix, posMax);
	
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	
	if (error != cudaSuccess) {
		printf("Error with kernel or d_matrix/d_trace allocation: %s\n", cudaGetErrorString(error));
		exit(0);
	}
	int *pos = (int*)malloc(1*sizeof(int));
	cudaMemcpy(pos, posMax, 1*sizeof(int), cudaMemcpyDeviceToHost);*/

	//Allocate and copy score table to host
	int *matrix =(int*)malloc(sizeof(int)*entries);
	cudaMemcpy(matrix, d_matrix, sizeof(int)*entries, cudaMemcpyDeviceToHost);
	
	//printf("Pos: %d\t%d\n", *pos, matrix[*pos]);
	
	//Allocate and copy trace table to host
	int *trace =(int*)malloc(sizeof(int)*entries);
	cudaMemcpy(trace, d_trace, sizeof(int)*entries, cudaMemcpyDeviceToHost);

	//Allocate final matrix: Used for output (easier printing)
	int *matrix2d = (int*)malloc(sizeof(int)*entries);
	CopyToMatrix(matrix2d, matrix, lenS1+1, lenS2+1);
	
	if (argc > 3 && !strcmp("-v",argv[3])){ 
		PrintMatrix(matrix2d, string1, string2, lenS1+1, lenS2+1);
	}

	//This Section causes an error:  "object was probably modified after being freed"
	/**Find largest value in matrix and then walk back until a '0' <- matrix[ix+j] value is found.
	 Find local alignment
	int max = 0;
	int maxPos = 0;
	for(i=0; i < (lenS1+1)*(lenS2+1);i++){
		if (matrix[i] > max){
			max = matrix[i];
			maxPos = i;
		}
	}

	int length = 0;
	i=maxPos/(lenS1+1) ;
	int j = maxPos%(lenS2+1);
	char *finalS1 = (char*)malloc(sizeof(i+j));
	char *finalS2 = (char*)malloc(sizeof(i+j));

	printf("max: %d\tmaxPos: %d\n", max, maxPos);
	while (matrix[i*(lenS2+1)+j] > 0){
		int dir = trace[i*(lenS2+1)+j];
		if (dir == -1){
			i--;
			finalS1[length] = string1[i];
			finalS2[length++] = '-';
		}else if (dir == 0){
			i--;j--;
			finalS1[length] = string1[i];
			finalS2[length++] = string2[j];
		}else{
			j--;
			finalS2[length] = string2[i];
			finalS1[length++] = '-';
		}
	}
	//printf("String1: %s\nString2: %s\n", finalS1, finalS2);
	for (i=length-1; i >= 0; i--){
		printf("%c",finalS1[i]);
	}
	printf("\n");
	for (i=length-1; i >= 0; i--){
		printf("%c",finalS2[i]);
	}
	free(finalS1);
	free(finalS2);
	*/

	//Free device memory
	cudaFree(d_string1);
	cudaFree(d_string2);
	cudaFree(d_matrix);
	cudaFree(d_trace);
	//cudaFree(posMax);
	
	//Free host memory
	free(string1);
	free(string2);
	free(matrix2d);
	free(matrix);
	free(trace);	

	cudaDeviceReset();
}

/**	
*
*/
void CalculateCost(int *d_matrix, int *d_trace, char *d_s1, char *d_s2, int s1, int s2){
	int i = 3;
	int prev = 1;
	int last = 0;

    for (int slice = 2; slice < s2 + s1 - 1; slice++) {
        int z1 = slice < s1 ? 0 : slice - s1 + 1;
        int z2 = slice < s2 ? 0 : slice - s2 + 1;
        
		int size = slice - z1 - z2 +1;
		int numElements = size -2;
		
		if (z2>1) last++;
		if (z1 > 0) numElements++;
		
		int off =1;
		if (z2 > 0) { numElements++; off = 0; };
		int blocksPerGrid = (numElements + numThreads - 1)/numThreads;
        
		ComputeDiagonal<<< blocksPerGrid, numThreads>>>
				(i + off, prev, last, numElements, d_matrix, d_trace, d_s1, d_s2, max(z2-1, 0), min(slice-2, s2-2));
		last = prev;
		prev = i;
		i += size;
    }
}

void PrintMatrix(int *arr, char *s1, char *s2, int xDim, int yDim){
	printf("\t");
	for(int i = 0; i < xDim - 1; i++){
		printf("\t%c", s1[i]);
	}
	printf("\n------------------------------------------------------------------------------------------------------\n\t|");
	for(int i =0; i < yDim; i++){	
		for(int j = 0; j < xDim; j++)
			printf("%d\t",arr[i*xDim + j]);
		printf("\n%c\t|", s2[i]);
	}printf("\n");
}

void CopyToMatrix(int *dst, int *src, int cols, int rows){
	/**Credit Mark Byers at Stack overflow: http://stackoverflow.com/a/2112951 */
	int i = 0;
	for (int slice = 0; slice < cols + rows - 1; ++slice) {
        int z1 = slice < cols ? 0 : slice - cols + 1;
        int z2 = slice < rows ? 0 : slice - rows + 1;
        for (int j = slice - z2; j >= z1; --j) {
            dst[cols*j + slice - j] = src[i++];
        }
    }
}


