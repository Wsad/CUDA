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

//Global Variables
int lenS1 =0;
int lenS2 = 0;

char *string1, *string2, *d_string1, *d_string2;

void CalculateCost(int *d_matrix, int *d_trace, int s1, int s2);

void CopyToMatrix(int *dst, int *src, int cols, int rows);

void PrintMatrix(int *arr, int xDim, int yDim);

//Kernel initializes all elements of matrix to 'value'
__global__ void init_matrix(int *matrix, int value, int maxElements){
	if (blockDim.x * blockIdx.x + threadIdx.x < maxElements)
		matrix[ blockDim.x * blockIdx.x + threadIdx.x] = value;
}


__global__ ComputeDiagonal(int i, int prevI, int lastI, int space, int *arr, int *trace, char *s1, char *s2, int s1off, int s2off){
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

	string1 = (char*)malloc(sizeof(char)*lenS1);
	string2 = (char*)malloc(sizeof(char)*lenS2);

	srand(time(NULL));
	int i;
	for(i=0; i<lenS1 ;i++)
		string1[i] = AGCT[rand()%4];
	for(i=0; i<lenS2 ;i++)
		string2[i] = AGCT[rand()%4];

	//printf("string1 %s\nstring2: %s\n", string1, string2);
	printf("\t|\t");
	for(int i =0; i < lenS1; i++)
		printf("%c\t",string1[i]);
	printf("\n");
	for(int i =0; i < 75; i++)
		printf("-");
	printf("\n");

	cudaMalloc((void**)&d_string1, sizeof(char)*lenS1);
	cudaMalloc((void**)&d_string2, sizeof(char)*lenS2);

	cudaMemcpy(d_string1, string1, sizeof(char)*lenS1, cudaMemcpyHostToDevice);
	cudaMemcpy(d_string2, string2, sizeof(char)*lenS2, cudaMemcpyHostToDevice);

	int entries = (lenS1+1)*(lenS2+1);

	//Initialize cost matrix on Device
	int* d_matrix;
	cudaMalloc((void**)&d_matrix, sizeof(int)*entries);

	init_matrix<<< entries/256+1, 256 >>>(d_matrix, 0, entries);

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
	CalculateCost(d_matrix, d_trace, lenS1+1, lenS2+1);
	cudaDeviceSynchronize();

	// Retrieve result from device and store it in host array
	cudaMemcpy(matrix, d_matrix, sizeof(int)*(lenS1+1)*(lenS2+1), cudaMemcpyDeviceToHost);
	cudaMemcpy(trace, d_trace, sizeof(int)*(lenS1+1)*(lenS2+1), cudaMemcpyDeviceToHost);

	CopyToMatrix(matrix2d, matrix, lenS1+1, lenS2+1);
	PrintMatrix(matrix2d,  lenS1+1, lenS2+1);

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
	free(matrix);
	cudaFree(d_matrix);

	free(trace);
	cudaFree(d_trace);

	free(string1);
	free(string2);
	cudaFree(d_string1);
	cudaFree(d_string2);
}

void CalculateCost(int *d_matrix, int *d_trace, int s1, int s2){
	int i = 3;
	int prev = 1;
	int last = 0;

    for (int slice = 2; slice < s2 + s1 - 1; slice++) {
        int z1 = slice < s1 ? 0 : slice - s1 + 1;
        int z2 = slice < s2 ? 0 : slice - s2 + 1;
        
		int size = slice - z1 - z2 +1;
		int threads = size -2;
		
		if (z2>1) last++;
		if (z1 > 0) threads++;
		
		int off =1;
		if (z2 > 0) { threads++; off = 0; };
        
		ComputeDiagonal<<<(threads/numThreads + 1), numThreads>>>
						(i + off, prev, last, threads, d_matrix, d_trace, d_string1, d_string2, max(z2-1, 0), min(slice-2, s2-2));
		last = prev;
		prev = i;
		i += size;
    }
}

void PrintMatrix(int *arr, int xDim, int yDim){
	printf("\t|");
	for(int i =0; i < yDim; i++){	
		for(int j = 0; j < xDim; j++)
			printf("%d\t",arr[i*xDim + j]);
		printf("\n%c\t|", string2[i]);
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


