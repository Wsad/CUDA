//#include "stdafx.h"
// Sequence Alignment -CUDA
// Alex Ringeri
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define numThreads 32

//Global Variables
int lenS1, lenS2;

char *string1, *string2, *d_string1, *d_string2;

void CalculateCostMatrix(int *d_matrix, int *d_trace);

void CopyToMatrix(int *dst, int *src, int cols, int rows);

void PrintMatrix(int *arr, int xDim, int yDim);

//Kernel initializes all elements of matrix to 'value'
__global__ void init_matrix(int *matrix, int value, int maxElements){
	if (blockDim.x * blockIdx.x + threadIdx.x < maxElements)
		matrix[ blockDim.x * blockIdx.x + threadIdx.x] = value;
}


__global__ void ComputeDiagonal(int index, int size, int *d_matrix, int *d_trace, char *d_s1, char *d_s2 , int offset1, int offset2){
	int threads = size + min(offset2*-1,0);
	if (blockIdx.x * blockDim.x + threadIdx.x < threads){
		int pos = index + blockIdx.x * blockDim.x + threadIdx.x;
		
		int up = d_matrix[pos - size + 1 + offset1];
		int left = d_matrix[pos - size + offset1];
		int upLeft = d_matrix[pos-2*size + offset2];
		
		if( d_s1[threadIdx.x] == d_s2[threads-threadIdx.x-1] )
			upLeft += 2;
		else
			upLeft -= 1;
		
		int cost;
		if (up > left)
			cost = up - 1;
		else
			cost = left - 1;
			
		if (upLeft > cost)
			cost = upLeft;
			
		d_matrix[pos] = max(cost,0);
		
		/** Debug
		d_matrix[pos - size + 1 + offset1] = pos - size + 1 + offset1;
		d_matrix[pos - size + offset1] = pos - size + offset1;
		d_matrix[pos - 2*size + offset2] = pos - 2*size + offset2;*/
		
	}
	else{
	}
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
	
	if (argc > 1){
		lenS1 = atoi(argv[1]);
		lenS2 = lenS1;
	}else {
		printf("No Command Line Arguments Set --- Exiting Program");
		exit(0);
	}
	
	string1 = (char*)malloc(sizeof(char)*lenS1);
	string2 = (char*)malloc(sizeof(char)*lenS2);

	srand(time(NULL));
	int i;
	/**To be changed: len2*/ 
	for(i=0; i<lenS2 ;i++){
		string1[i] = AGCT[rand()%4];
		string2[i] = AGCT[rand()%4];
	}

	//printf("string1 %s\nstring2: %s\n", string1, string2);	
	/*printf("\t");	
	for(int i =0; i < lenS1; i++)
		printf("%c\t",string1[i]);
	printf("\n\t");
	for(int i =0; i < lenS2; i++)
		printf("%c\t",string2[i]);
	printf("\n");
	for(int i =0; i < 100; i++)
		printf("-");
	printf("\n");*/
	
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
	init_matrix<<< ceilf(((float)entries)/256), 256 >>>(d_trace, -2, entries);
	
	//Allocate and copy trace matrix to Host
	int *trace =(int*)malloc(sizeof(int)*entries);
	cudaMemcpy(trace, d_trace, sizeof(int)*entries, cudaMemcpyDeviceToHost);

	//Allocate final matrix: Used for output (easier printing)
	int *matrix2d = (int*)malloc(sizeof(int)*entries);
	
	// Do calculation on device:
	CalculateCostMatrix(d_matrix, d_trace);
	cudaDeviceSynchronize();
	
	// Retrieve result from device and store it in host array
	cudaMemcpy(matrix, d_matrix, sizeof(int)*(lenS1+1)*(lenS2+1), cudaMemcpyDeviceToHost);
	cudaMemcpy(trace, d_trace, sizeof(int)*(lenS1+1)*(lenS2+1), cudaMemcpyDeviceToHost);
	
	CopyToMatrix(matrix2d, matrix, lenS1+1, lenS2+1);
	//PrintMatrix(matrix2d,  lenS1+1, lenS2+1);
	
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

	// Cleanup
	free(matrix2d);
	cudaFree(d_matrix);

	free(trace);
	cudaFree(d_trace);
	
	free(string1); 
	free(string2); 
	cudaFree(d_string1); 
	cudaFree(d_string2);
}

void CalculateCostMatrix( int *d_matrix, int *d_trace ){
	//Top section of cost table
	int size=3; int index = 3;
	int numBlocks =1; /*int numThreads;*/ int row;
	for (row=2; row <= lenS2; row++){
		//numThreads = size - 2;
		numBlocks = (size-2)/numThreads +1;
		ComputeDiagonal<<< numBlocks, numThreads >>>(index+1, size, d_matrix, d_trace, d_string1, d_string2 , 0, 2);
		index += size;
		size++;
	}
	size--;
	
	//Middle section of cost table
	int k = lenS1 - lenS2 + 1;
	if (k > 1){
		//numThreads = size - 1;
		numBlocks = (size-1)/numThreads +1;
	  	ComputeDiagonal<<< numBlocks, numThreads >>>(index, size, d_matrix, d_trace, d_string1, d_string2, 0, 1);
		index += size;
	  	for (int i = k ; i > 2; i--){
	    	ComputeDiagonal<<< numBlocks, numThreads >>>(index, size, d_matrix, d_trace, d_string1, d_string2, 0, 1);
			index += size;
	    	row++;
	    }
	  	size--;
		ComputeDiagonal<<< numBlocks, numThreads >>>(index, size, d_matrix, d_trace, d_string1, d_string2, -1, -1);
		index += size;
	  	row += 2;
	}else{
	  	size--;
	  	//numThreads = size;
	  	ComputeDiagonal<<< numBlocks, numThreads >>>(index, size, d_matrix, d_trace, d_string1, d_string2, -1, -1);
		index += size; 
		row++;
	}
	
	//Bottom section of cost table
	for(int r = row; r < lenS1+lenS2+1; r++){
		size--;
		//numThreads = size;
		ComputeDiagonal<<< numBlocks, numThreads >>>(index, size, d_matrix, d_trace, d_string1, d_string2, -1, -2);
		index += size;
	}
}

void PrintMatrix(int *arr, int xDim, int yDim){
	for(int i =0; i < yDim; i++){
		for(int j = 0; j < xDim; j++)
			printf("%d\t",arr[i*xDim + j]);
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
