// Sequence Alignment -CUDA
// Alex Ringeri

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int numThreads = 256;

void CalculateCost(int *d_matrix, int *d_trace, char *d_s1, char *d_s2, int s1, int s2, int comparison, int entries);

void CopyToMatrix(int *dst, int *src, int cols, int rows);

void PrintMatrix(int *arr, char *s1, char *s2, int xDim, int yDim);

//Kernel initializes all elements of matrix to 'value'
__global__ void init_matrix(int *matrix, int value, int maxElements) {
	if (blockDim.x * blockIdx.x + threadIdx.x < maxElements)
		matrix[blockDim.x * blockIdx.x + threadIdx.x] = value;
}

__global__ void ComputeDiagonal(int i, int prevI, int lastI, int space, int *arr, int *trace, char *s1, char *s2, int s1off, int s2off) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < space) {
		int left = arr[prevI + id];
		int up = arr[prevI + id + 1];
		int upLeft = arr[lastI + id];

		if (s1[s1off + id] == s2[s2off - id])
			upLeft += 2;
		else
			upLeft -= 1;

		int cost, dir;
		if (up > left) {
			cost = up - 1;
			dir = 1;
		} else {
			cost = left - 1;
			dir = -1;
		}
		if (upLeft > cost) {
			cost = upLeft;
			dir = 0;
		}
		arr[i + id] = max(cost, 0);
		trace[i + id] = dir;
	}
}

__global__ void CalculateTiledMatrix(int *d_m, int *d_t, char *d_s1, char *d_s2, int cols, int i, int entries, int blockNum){
	extern __shared__ int tile[];
	int *sh_m = &tile[0]; 								//pointer to matrix in shared memory
	char *sh_s1 = (char*)&tile[(blockDim.x + 1 )*(blockDim.x + 1)];			//pointer to 1st string in shared memory
	char *sh_s2 = (char*)&tile[(blockDim.x + 1 )*(blockDim.x + 1) + blockDim.x]; 	//pointer to 2nd string in shared memory
	//Copy boundary data into shared memory
	int id = blockIdx.x*blockDim.x + threadIdx.x;	
	int index = i - (blockIdx.x*blockDim.x)*cols + id;

	// Todo: Check condition, blocks larger thread numbers when copying border data to shared memory (rectangular matrices)
	if (/*i%cols + id - 1 < cols &&*/ i-(blockIdx.x)*(blockDim.x)*cols -1 + threadIdx.x*cols < entries){/*index + threadIdx.x*cols -1*/
		if (threadIdx.x == 0){
			sh_m[0] = d_m[index - cols -1];
		}
		//sh_m[threadIdx.x + 1] = d_m[index - cols];
		sh_s2[threadIdx.x] = d_s2[i/cols-(blockIdx.x)*(blockDim.x) -1 + threadIdx.x];
		/*if (blockIdx.x == 1)		
			printf("index: %d\ts2: %d\t\n",index,i/cols-(blockIdx.x)*(blockDim.x) -1 + threadIdx.x);*/
	}
	if (i%cols + id  < cols /*&& i%cols + id - 1 < entries*/ ){
		sh_s1[threadIdx.x] = d_s1[i%cols + id - 1];
		sh_m[threadIdx.x + 1] = d_m[index - cols];
		/*if (blockIdx.x == 1)		
			printf("index: %d\tS1 %d\n",index, i%cols + id -1);*/
	}
	if (i - blockIdx.x*blockDim.x*cols + blockIdx.x*blockDim.x + cols*threadIdx.x - 1 < entries){
		sh_m[(blockDim.x + 1)*(threadIdx.x + 1)] = d_m[i - blockIdx.x*blockDim.x*cols + blockIdx.x*blockDim.x + cols*threadIdx.x - 1 ];
		//printf("s1[%d]\n", index%cols - 1);
	}
	syncthreads(); 

	//Calculate cost for each diagonal
	for (int b = 0; b < 2; b++){
		for(int d = 1; d < blockDim.x + 1; d++){
			if (threadIdx.x < blockDim.x*b - b*d + d*(1-b) ){
				int upLeft = sh_m[(d - d*b + blockDim.x*b - threadIdx.x - 1)*(blockDim.x+1) + d*b + threadIdx.x];//[(d - threadIdx.x - 1)*(blockDim.x+1) + threadIdx.x]
				int up = sh_m[(d - d*b + blockDim.x*b - threadIdx.x - 1)*(blockDim.x+1) + threadIdx.x + d*b + 1];//[(d - threadIdx.x - 1)*(blockDim.x+1) + threadIdx.x + 1]
				int left = sh_m[(d - d*b + blockDim.x*b - threadIdx.x)*(blockDim.x+1) + d*b + threadIdx.x];//[(d - threadIdx.x)*(blockDim.x+1) + threadIdx.x]

				if (sh_s1[threadIdx.x + d*b] == sh_s2[d - d*b + blockDim.x*b - 1-threadIdx.x])
					upLeft += 2;
				else
					upLeft -= 1;

				int cost; //int dir;
				if (up > left) {
					cost = up - 1;
					//dir = 1;
				} else {
					cost = left - 1;
					//dir = -1;
				}
				if (upLeft > cost) {
					cost = upLeft;
					//dir = 0;
				}
				sh_m[(d -d*b + blockDim.x*b - threadIdx.x)*(blockDim.x+1) + d*b + threadIdx.x + 1] = max(0, cost);
				/*if (threadIdx.x == 31 && index > 79){
					printf("ID: %d\tS1 %c\tS2 %c\tUpleft %d\tLeft %d\tUp %d\tCost: %d\ts1[%d]\ts2[%d]\n", index, sh_s1[threadIdx.x+d*b], sh_s2[d - d*b + blockDim.x*b - 1-threadIdx.x], upLeft, left, up, max(0,cost),threadIdx.x +d*b,d - d*b + blockDim.x*b - 1-threadIdx.x );
					//printf("index: %d\ts2: %d\n",index,index/cols+threadIdx.x -1);
				}*/
			}
			syncthreads();
		}
	}

	//Copy results from shared memory into global memory
	for (int j = 0; j < blockDim.x; j++){
		if (i%cols + id < cols && index + j*cols < entries){
			d_m[index + j*(cols)] = sh_m[(j+1)*(blockDim.x+1) + threadIdx.x + 1];
		}
	}
}

__global__ void CalculateTiledDiagonal(int *d_matrix, int *d_trace, char *d_s1, char *d_s2, int diag, int rows, int cols) {

	int dSize = 1;

	for(int d = diag; d < 2*blockDim.x-1+diag; d++){
		int z1 = 0;
		int z2 = 0;
		int index = 1;
		int size = 0;
		if (d > rows){
			z1 = d - rows;
			index+= blockDim.x - 1;
		}
		if (d > cols){
			z2 = d - cols;
			size++;
			index--;
		}
		//int z2 = (diag - cols < 0)? 0 : diag - rows;
		index += (d*(d+1) + z1*(z1+1) + z2*(z2+1))/2;
		//int size += diag - z1 - z2 + 1;
		int pos = blockIdx.x * blockDim.x + threadIdx.x;
		if (d > blockDim.x -1 + diag)
			index += d - diag - blockDim.x +1;
		if (threadIdx.x < dSize){
			d_matrix[index + pos] = index + pos;
			//printf("Block ID : %d Thread ID: %d, pos : %d index : %d d: %d\n",blockIdx.x, threadIdx.x, pos, index, d);
		}

		dSize++;
	}

	//loop through diagonals and copy into shared memory
	//copy strings into shared memory

	//loop through diagonals in shmem and calculate costs

	//write result from shmem to global memory
}

__global__ void CalculateCostOneKernel(int *d_matrix, int *d_trace, char *d_s1, char *d_s2, int s1, int s2) {
	int i = 3;
	int prev = 1;
	int last = 0;

	for (int slice = 2; slice < s2 + s1 - 1; slice++) {
		int z1 = 0;
		int z2 = 0;
		int numElements = 0;
		int off = 1;
		if (slice > s1 - 1) {
			z1 = slice - s1 + 1;
			numElements++;
		}
		if (slice > s2 - 1) {
			z2 = slice - s2 + 1;
			numElements++;
			off = 0;
		}
		int size = slice - z1 - z2 + 1;
		numElements += size - 2;
		if (z2 > 1)
			last++;

		for (int s = 0; s < (numElements + blockDim.x - 1) / blockDim.x; s++) {
			int id = blockDim.x * s + threadIdx.x;
			if (id < numElements) {
				int upLeft = d_matrix[last + id];
				int left = d_matrix[prev + id];
				int up = d_matrix[prev + id + 1];

				if (d_s1[max(z2 - 1, 0) + id] == d_s2[min(slice - 2, s2 - 2) - id])
					upLeft += 2;
				else
					upLeft -= 1;

				int cost, dir;
				if (up > left) {
					cost = up - 1;
					dir = 1;
				} else {
					cost = left - 1;
					dir = -1;
				}
				if (upLeft > cost) {
					cost = upLeft;
					dir = 0;
				}
				d_matrix[i + off + id] = max(cost, 0);
				d_trace[i + off + id] = dir;
			}
		}
		last = prev;
		prev = i;
		i += size;
		syncthreads();
	}
}

// Main routine:
int main(int argc, char *argv[]) {
	cudaSetDevice(0);
	char AGCT[] = "AGCT";
	int lenS1, lenS2;
	int numComparisons = 0;
	int approach = 0;
	if (argc > 4) {
		int args[] = { atoi(argv[1]), atoi(argv[2]) };
		if (args[0] > args[1]) {
			lenS1 = args[0];
			lenS2 = args[1];
		} else {
			lenS1 = args[1];
			lenS2 = args[0];
		}
		numComparisons = atoi(argv[3]);
		approach = atoi(argv[4]);
		if (approach < 1 || approach > 3) {
			printf("Invalid Approach Argument --- Exiting Program\n");
			exit(0);
		}
	}
	else {
		printf("Invalid Command Line Arguments --- Exiting Program\n");
		exit(0);
	}

	printf("Calculating Cost Matrix: %d elements (%d x %d)\n", (lenS1 + 1) * (lenS2 + 1), lenS1 + 1, lenS2 + 1);

	//Allocate strings on host
	char * string1 = (char*) malloc(sizeof(char) * lenS1);
	char * s2Arr = (char*) malloc(sizeof(char) * numComparisons * lenS2);

	//Initialize strings with random numbers
	srand(time(NULL));
	for (int i = 0; i < lenS1; i++) {
		char r = AGCT[rand() % 4];
		string1[i] = r;
	}
	for (int i = 0; i < numComparisons; i++) {
		for (int j = 0; j < lenS2; j++) {
			char r = AGCT[rand() % 4];
			s2Arr[i * lenS2 + j] = r;
		}
	}
	//Allocate strings on device
	cudaError_t error = cudaSuccess;
	char *d_string1, *d_s2Arr;
	int *d_matrixArr;
	unsigned int entries = (lenS1 + 1) * (lenS2 + 1);

	error = cudaMalloc((void**) &d_string1, sizeof(char) * lenS1);
	if (error != cudaSuccess) {
		printf("Error allocating s1 on device\n");
		cudaDeviceReset(); exit(0);
	}

	error = cudaMalloc((void**) &d_s2Arr, sizeof(char) * numComparisons * lenS2);
	if (error != cudaSuccess) {
		printf("Error allocating s2array on device\n");
		cudaDeviceReset(); exit(0);
	}

	//Initialize sequence strings on device
	error = cudaMemcpy(d_string1, string1, sizeof(char) * lenS1, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		printf("Error copying k1s1 to device\n");
		cudaDeviceReset(); exit(0);
	}

	for (int i = 0; i < numComparisons; i++) {
		error = cudaMemcpy(d_s2Arr, s2Arr, sizeof(char) * numComparisons * lenS2, cudaMemcpyHostToDevice);
		if (error != cudaSuccess) {
			printf("Error copying a s2Arr to s_S2arr");
			cudaDeviceReset(); exit(0);
		}
	}

	/****	 	Allocate cost matrix 		****/
	error = cudaMalloc((void**) &d_matrixArr, sizeof(int) * numComparisons * entries);
	if (error != cudaSuccess) {
		printf("Error allocating d_matrixArr on device\n");
		cudaDeviceReset(); exit(0);
	}

	//Allocate trace table on Device
	int *d_trace;
	error = cudaMalloc((void**) &d_trace, sizeof(int) * entries);
	if (error != cudaSuccess) {
		printf("Error allocating k1 d_trace on device:\n%s", cudaGetErrorString(error));
		cudaDeviceReset(); exit(0);
	}

	//Initialize trace and score tables
	int threadsPerBlock = 1024;
	int blocksPerGrid = (entries + threadsPerBlock - 1) / threadsPerBlock;
	for(int i=0; i < numComparisons; i++)
		init_matrix<<<blocksPerGrid, threadsPerBlock>>>(&d_matrixArr[i*entries], 0, entries);
	init_matrix<<<blocksPerGrid, threadsPerBlock>>>(d_trace, -2, entries);
	cudaDeviceSynchronize();

	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf( "Error initializing arrays on device(Kernel Launch: init_matrix)\n%s\n", cudaGetErrorString(error));
		exit(0);
	}

	/* Do calculation on device: */
	if (approach == 1) {
		//Calculate seperate problems concurrently over many streams (one kernel per problem)
		cudaStream_t s[numComparisons];
		for (int i = 0; i < numComparisons; i++) {
			cudaStreamCreate(&s[i]);
			CalculateCostOneKernel<<<1, 256, 0, s[i]>>>(&d_matrixArr[i * entries], d_trace, d_string1, &d_s2Arr[i*lenS2], lenS1 + 1, lenS2 + 1);
		}
		for (int i = 0; i < numComparisons; i++)
			cudaStreamDestroy(s[i]);
	} 
	else if (approach == 2) {
		//Calculate one problem with many kernels
		CalculateCost(d_matrixArr, d_trace, d_string1, d_s2Arr, lenS1 + 1,lenS2 + 1, numComparisons, entries);
	} 
	else {

		CalculateCost(d_matrixArr, d_trace, d_string1, d_s2Arr, lenS1+1, lenS2+1, numComparisons, entries);
		cudaDeviceSynchronize();
		int *a = (int*)malloc(sizeof(int)*entries);
		cudaMemcpy(a, d_matrixArr, sizeof(int)*entries, cudaMemcpyDeviceToHost);

		threadsPerBlock = 1024;
		blocksPerGrid = (entries + threadsPerBlock - 1) / threadsPerBlock;
		init_matrix<<< blocksPerGrid, threadsPerBlock >>>(d_matrixArr, 0, entries);

		cudaDeviceSynchronize();
		int *m2d = (int*) malloc(sizeof(int) * entries);
		CopyToMatrix(m2d, a, lenS1+1, lenS2+1);
		//PrintMatrix(m2d, string1, s2Arr, lenS1+1, lenS2+1);


		
		threadsPerBlock = 32;
		blocksPerGrid = 0;
		int blocksInRow = (lenS1 + threadsPerBlock -1)/threadsPerBlock;
		int blocksInCol = (lenS2 + threadsPerBlock -1)/threadsPerBlock;
		int index = lenS1 + 2;
		for(int i =0; i < blocksInRow + blocksInCol - 1; i++){		
			int z1 = 0; int z2 = 0;		
			if (i >= blocksInRow)
				z1 = i - blocksInRow +1;
			if (i >= blocksInCol)
				z2 = i - blocksInCol +1;	

			if (i == 0){
				blocksPerGrid++;
			}			
			else if (z1 == 0 && z2 == 0){
				index += threadsPerBlock*(lenS1+1);
				blocksPerGrid++;
			}
			else if (z1 > 0 && z2 > 0){
				index += threadsPerBlock;
				blocksPerGrid--;
			}
			else{
				index += threadsPerBlock;
			}
			CalculateTiledMatrix<<< blocksPerGrid, threadsPerBlock, sizeof(int)*(threadsPerBlock+1)*(threadsPerBlock+1) + sizeof(char)*2*threadsPerBlock >>>(d_matrixArr, d_trace, d_string1, d_s2Arr, (lenS1+1), index, entries, blocksPerGrid);
		}	

		cudaDeviceSynchronize();
		int *r = (int*)malloc(sizeof(int)*entries);
		cudaMemcpy(r, d_matrixArr, sizeof(int)*entries, cudaMemcpyDeviceToHost);
		//PrintMatrix(r, string1, s2Arr, lenS1+1, lenS2+1);		
		/*for (int j =28; j < lenS1; j++)
			printf(" |\t%c", s2Arr[j]);
		printf("\n\t");
		for(int i = 0; i < lenS2 +1; i++){
			for( int j =29; j <lenS1 +1; j++)
				printf("%d\t", r[i*(lenS1+1) + j]);
			printf("\n%c | \t", s2Arr[i]);
		}
				printf("\n\n");
		for (int j =28; j < lenS1; j++)
			printf(" |\t%c", s2Arr[j]);
		printf("\n\t");
		for(int i = 0; i < lenS2 +1 ; i++){
			for( int j =29; j <lenS1 +1; j++)
				printf("%d\t", m2d[i*(lenS1+1) + j]);
			printf("\n%c | \t", s2Arr[i]);
		}*/

		for(int n = 0; n < entries; n++){
			if (m2d[n] != r[n]){
				printf("Tiled result different from Diagonal Kernel result: index %d\n", n);
				free(m2d); free(r); free(a);
				cudaDeviceReset();
				exit(0);
			}
		}
		printf("Tiled result same as Diagonal\n");
		free(r); free(a); free(m2d);
		/*//Calculate cost using tiled method
		threadsPerBlock = 256;
		cudaStream_t s[numComparisons];
		for (int i = 0; i < numComparisons; i++)
			cudaStreamCreate(&s[i]);
		for (int diag = 2; diag < (lenS1 + lenS2 + 1); diag += threadsPerBlock) {
			for (int c = 0; c < numComparisons; c++) {
				//Launch Tiled Kernel here				
				threadsPerBlock = 5;
				blocksPerGrid = c+1;
				printf("Kernel c=%d, d=%d\n", c,diag);
				CalculateTiledDiagonal<<< blocksPerGrid, threadsPerBlock, 0, s[c]>>>(&d_matrixArr[c*entries], d_trace, d_string1, &d_s2Arr[c*lenS2], diag, lenS1+1, lenS2+1);
				//printf("after\n");			
			}
		}
		for (int i = 0; i < numComparisons; i++)
			cudaStreamDestroy(s[i]);*/
	}
	cudaDeviceSynchronize();


	error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("Error with kernel launches:(calculating costs) %s\n", cudaGetErrorString(error));
		cudaDeviceReset(); exit(0);
	}

	//Allocate and copy score table to host
	int *k1Result = (int*) malloc(sizeof(int) * entries);

	//Allocate final matrix: Used for output (easier printing)
	int *matrix2d = (int*) malloc(sizeof(int) * entries);

	for (int a = 0; a < numComparisons; a++) {
		cudaMemcpy(k1Result, &d_matrixArr[a * entries], sizeof(int) * entries, cudaMemcpyDeviceToHost);
		CopyToMatrix(matrix2d, k1Result, lenS1 + 1, lenS2 + 1);

		if (argc > 5 && !strcmp("-v", argv[5])) {
			printf("Kernel %d:\n", a);
			PrintMatrix(matrix2d, string1, &s2Arr[a * lenS2], lenS1 + 1,lenS2 + 1);
			
		}
	}
	//Allocate and copy trace table to host
	//CopyToMatrix(matrix2d, trace, lenS1+1, lenS2+1);
	cudaDeviceSynchronize();

	//Free device memory
	cudaFree(d_string1);
	cudaFree(d_trace);
	cudaFree(d_s2Arr);
	cudaFree(d_matrixArr);

	//Free host memory
	free(string1);
	free(matrix2d);
	free(k1Result);
	free(s2Arr);

	cudaDeviceReset();
	printf("Calculation complete\n");
}

/**	
 *	Method launches one kernel per diagonal to calculate matrix
 */
void CalculateCost(int *d_matrix, int *d_trace, char *d_s1, char *d_s2, int s1, int s2, int comparisons, int entries) {
	int i = 3;
	int prev = 1;
	int last = 0;

	cudaStream_t stream[comparisons];
	for (int a = 0; a < comparisons; a++)
		cudaStreamCreate(&stream[a]);

	for (int slice = 2; slice < s2 + s1 - 1; slice++) {
		int z1 = slice < s1 ? 0 : slice - s1 + 1;
		int z2 = slice < s2 ? 0 : slice - s2 + 1;

		int size = slice - z1 - z2 + 1;
		int numElements = size - 2;

		if (z2 > 1)
			last++;
		if (z1 > 0)
			numElements++;

		int off = 1;
		if (z2 > 0) {
			numElements++;
			off = 0;
		};
		int blocksPerGrid = (numElements + numThreads - 1) / numThreads;

		for (int a = 0; a < comparisons; a++)
			ComputeDiagonal<<<blocksPerGrid, numThreads, 0, stream[a]>>>(i + off, prev, last, numElements, &d_matrix[a * entries], d_trace, d_s1, &d_s2[a*(s2-1)], max(z2 - 1, 0), min(slice - 2, s2 - 2));

		last = prev;
		prev = i;
		i += size;
	}
	for (int a = 0; a < comparisons; a++)
		cudaStreamDestroy(stream[a]);

}

void PrintMatrix(int *arr, char *s1, char *s2, int xDim, int yDim) {
	printf("\t");
	for (int i = 0; i < xDim - 1; i++) {
		printf("\t%c", s1[i]);
	}
	printf("\n------------------------------------------------------------------------------------------------------\n\t|");
	for (int i = 0; i < yDim; i++) {
		for (int j = 0; j < xDim; j++)
			printf("%d\t", arr[i * xDim + j]);
		(i == yDim - 1) ? printf("\n") : printf("\n%c\t|", s2[i]);
	}
	printf("\n");
}

void CopyToMatrix(int *dst, int *src, int cols, int rows) {
	/**Credit Mark Byers at Stack overflow: http://stackoverflow.com/a/2112951 */
	int i = 0;
	for (int slice = 0; slice < cols + rows - 1; ++slice) {
		int z1 = slice < cols ? 0 : slice - cols + 1;
		int z2 = slice < rows ? 0 : slice - rows + 1;
		for (int j = slice - z2; j >= z1; --j) {
			dst[cols * j + slice - j] = src[i++];
		}
	}
}

