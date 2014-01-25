//#include "stdafx.h"
// Sequence Alignment -CUDA
// Alex Ringeri
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

void CopyToMatrix(int *dst, int *src, int cols, int rows);

void PrintMatrix(int *arr, int xDim, int yDim);

//Kernel initializes all elements of matrix to 'value'
__global__ void init_matrix(int *matrix, int value, int maxElements){
	if (blockDim.x * blockIdx.x + threadIDx.x < maxElements)
		matrix[ blockDim.x * blockIdx.x + threadIDx.x] = value;
}
 
// Kernel that executes on the CUDA device
__global__ void calc_matrix(int xPos, int yPos, int *matrix, int *result, int colSize, char* s1, char* s2){
	int x = xPos + threadIdx.x;
	int y = yPos - threadIdx.x;
	//printf("ThreadID: %d\tx: %d\ty: %d\t#:%d\n",threadIdx.x,x,y,(x*colSize+y));
	int up = matrix[x*colSize+y-1];
	int left = matrix[(x-1)*colSize+y];
	int upLeft = matrix[(x-1)*colSize+y-1];
	int match, trace = -2;
	if (s1[x-1] == s2[y-1])
		match = 2;
	else
		match = -1;
	int maxV = max( left -1, up - 1);
	if (left -1 > up -1){
		maxV = left-1;
		trace = -1;
	} else {
		maxV = up -1;
		trace = 1;
	} if ( upLeft + match > maxV){
		maxV = upLeft+match;
		trace = 0;
	}
	//store cost in 'matrix'
	matrix[x*colSize+y] = max(maxV,0);
	//store choice in result matrix. For use when retracing at end.
	result[x*colSize+y] = trace;
	//matrix[x*colSize+y] = x*colSize+y;
	
}

//Kernel Using Shared Memory
__global__ void calc_shm( int tileSize, int row, int rank, int *matrix, int *result, char* s1, char* s2){
	extern __shared__ int tile[];
	int id = threadIdx.x;
	int pos = rank + blockIdx.x * blockDim.x + id;
	
	//Load neighbors into shared memory
	tile[id] = matrix[pos-2*row];//upleft from matrix
	tile[row+id-2] = matrix[pos-row-1]; //left from matrix
	tile[row + id - 1] = matrix[pos-row];//up from matrix
	
	syncthreads();
	int up = tile[row + id - 1];
	int left = tile[row + id -2];
	int upLeft = tile[id];
	
	/**Load neighbors using Global memory
	int up = matrix[pos-row];
	int left = matrix[pos-row-1];
	int upLeft = matrix[pos-2*row];*/
	
	//printf("ThreadID: %d\ts1: %c\ts2: %c\n",threadIdx.x,s1[id],s2[row-2-id]);
	//printf("ThreadID: %d\tx: %d\ty: %d\t#:%d\n",threadIdx.x,x,y,(x*colSize+y));
	
	int match, trace = -2;
	if (s1[id] == s2[row-2-id])
		match = 2;
	else
		match = -1;
	int maxV = max( left -1, up - 1);
	if (left -1 > up -1){
		maxV = left-1;
		trace = -1;
	} else {
		maxV = up -1;
		trace = 1;
	} if ( upLeft + match > maxV){
		maxV = upLeft+match;
		trace = 0;
	}
	//store cost in 'd_matrix'
	matrix[pos] = max(maxV,0);
	//printf("ThreadID: %d\tup: %d\tleft: %d\tupleft: %d\tMax: %d\trank: %d\trow: %d\n",threadIdx.x,up,left,upLeft,maxV, pos, row);
	
	//store choice in 'd_trace' matrix. For use when retracing at end.
	result[pos] = trace;
}
 
// Main routine: Executes on the host
int main(int argc, char *argv[]){
	char AGCT[] = "AGCT";
	int lenS1=0;
	
	printf("Enter size: ");	
	scanf("%d",&lenS1);
	int lenS2 = lenS1;
	
	char *string1 = (char*)malloc(sizeof(char)*lenS1);
	char *string2 = (char*)malloc(sizeof(char)*lenS2);

	srand(time(NULL));
	int i;
	/**To be changed: len2*/ 
	for(i=0; i<lenS2 ;i++){
		string1[i] = AGCT[rand()%4];
		string2[i] = AGCT[rand()%4];
	}

	printf("string1 %s\nstring2: %s\n", string1, string2);	

	char * d_string1;
	char * d_string2;
	cudaMalloc((void**)&d_string1, sizeof(char)*lenS1);
	cudaMalloc((void**)&d_string2, sizeof(char)*lenS2);
	
	cudaMemcpy(d_string1, string1, sizeof(char)*lenS1, cudaMemcpyHostToDevice);
	cudaMemcpy(d_string2, string2, sizeof(char)*lenS2, cudaMemcpyHostToDevice);
	
	int entries = (lenS1+1)*(lenS2+1);
	
	//Initialize cost matrix on Device
	int* d_matrix;
	cudaMalloc((void**)&d_matrix, sizeof(int)*entries);
	
	init_matrix<<< ceilf(((float)entries)/256), 256 >>>(d_matrix, 0, entries);
	
	//Allocate and copy cost matrix to Host
	int *matrix =(int*)malloc(sizeof(int)*entries);
	cudaMemcpy(matrix, d_matrix, sizeof(int)*entries, cudaMemcpyDeviceToHost);
	
	//Initialize trace matrix on Device
	int* d_trace;
	cudaMalloc((void**)&d_trace, sizeof(int)*entries);
	init_matrix<<< ceilf(((float)entries)/256), 256 >>>(trace, -2, entries);
	
	//Allocate and copy trace matrix to Host
	int *trace =(int*)malloc(sizeof(int)*entries);
	cudaMemcpy(trace, d_trace, sizeof(int)*entries, cudaMemcpyDeviceToHost);
	
	
	
	
	
	

	
	//Allocate final matrix: Used for output (easier printing)
	int *matrix2d = (int*)malloc(sizeof(int)*entries);
	
	// Do calculation on device: Assume square matrix for now
	int maxThreads = min(lenS1,lenS2);
	int size=3; int index = 4;
	for (i=1; i < lenS2; i++){
		calc_shm<<< 1, i, size*sizeof(int) >>>(size, i+1, index, d_matrix, d_trace, d_string1, d_string2);
		size += 2;
		index += i + 2;
		cudaDeviceSynchronize();
		/*cudaMemcpy(matrix, d_matrix, sizeof(int)*(lenS1+1)*(lenS2+1), cudaMemcpyDeviceToHost);
		CopyToMatrix(matrix2d,matrix,lenS1+1,lenS2+1);
		PrintMatrix(matrix2d, lenS1+1, lenS2+1);*/
	}

	
	/* original kernel
	for(i=1; i < maxThreads; i++){
		calc_matrix <<< 1, i >>> (1, i, d_matrix, d_trace, (lenS2+1), d_string1, d_string2);
		cudaDeviceSynchronize();
	}
	for(i = maxThreads; i > 0; i--){
		//printf("%d\n",i);
		calc_matrix <<< 1, i >>>(maxThreads-i+1, lenS2, d_matrix, d_trace, (lenS2+1), d_string1, d_string2);
		cudaDeviceSynchronize();
	}*/
	
	// Retrieve result from device and store it in host array
	cudaMemcpy(matrix, d_matrix, sizeof(int)*(lenS1+1)*(lenS2+1), cudaMemcpyDeviceToHost);
	cudaMemcpy(trace, d_trace, sizeof(int)*(lenS1+1)*(lenS2+1), cudaMemcpyDeviceToHost);
	
	
	/* Print results from Original Kernel
	printf("%d\t", matrix[0]);
	i=0;j=2;
	while(i < entries){
		i += j;
		printf("%d\t", matrix[i]);
		j++
	}
		
	}
	j = 0;
	for( i =0; i <= lenS2; i++){
		for(j=i; j <=lenS2; j++){
			printf("%d\t" matrix[((j+2)*(j+1))/2 - i -1]);
			if ((j+1)%lenS2 == 0)
				printf("\n");
		}
	}*/
	/*
	printf("Cost Matrix:\n");
	for (i=0; i <=lenS2; i++) {
		for (int j=0; j<=lenS1;j++){
			printf("%d\t", matrix[j*(lenS2+1)+i]);
		}
		printf("\n");
	}
	printf("\nTrace Matrix:\n");
	for (i=0; i <=lenS2; i++) {
		for (int j=0; j<=lenS1;j++){
			printf("%d\t", trace[j*(lenS2+1)+i]);
		}
		printf("\n");
	}*/
	

	/**Find largest value in matrix and then walk back until a '0' <- matrix[ix+j] value is found.
	 Find local alignment*/
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
	printf("\n");
	// Cleanup
	
	free(matrix2d);
	cudaFree(d_matrix);

	free(trace);
	cudaFree(d_trace);
	
	free(string1); 
	free(string2); 
	cudaFree(d_string1); 
	cudaFree(d_string2);
	
	free(finalS1);
	free(finalS2);
}

void PrintMatrix(int *arr, int xDim, int yDim){
	for(int i =0; i < yDim; i++){
		for(int j = 0; j < xDim; j++)
			printf("%d ",arr[i*xDim + j]);
		printf("\n");
	}
}

void CopyToMatrix(int *dst, int *src, int cols, int rows){
	//Credit: Jan on Stackoverflow
	// traverse array diagonally
	int c, tmp, x, i;
	i=0;
	for (c = cols - 1; c > -cols; c--) {
		tmp = cols - abs(c) - 1;
		x = tmp;
		while (x >= 0) {
			if (c >= 0) {
				dst[x*cols +(tmp - x)] = src[i++];
			}
			else {
				dst[(cols - (tmp - x) - 1)*cols + ((cols-1)-x)] = src[i++];
			}
			--x;
		}
		//std::cout << "\n";
	}
}