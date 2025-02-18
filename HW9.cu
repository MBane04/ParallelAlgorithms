// Name: Mason Bane
// Vector Dot product on many block 
// nvcc HW9.cu -o temp
/*
 What to do:
 This code is the solution to HW8. It finds the dot product of vectors that are smaller than the block size.
 Extend this code so that it uses many blocks and many threads and can find the dot product of any vector length.
 Use shared memory in your blocks to speed up your code.
 You will have to do the final reduction on the CPU.
 Set your thread count to 200. Set N to different values to check your code.
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define N 15'176// Length of the vector; works with 15'000, 15'001, 22

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
float DotCPU, DotGPU;
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.01;

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void allocateMemory();
void innitialize();
void dotProductCPU(float*, float*, int);
__global__ void dotProductGPU(float*, float*, float*, int);
bool  check(float, float, float);
long elaspedTime(struct timeval, struct timeval);
void cleanUp();

// This check to see if an error happened in your CUDA code. It tell you what it thinks went wrong,
// and what file and line it occured on.
void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

// This will be the layout of the parallel space we will be using.
void setUpDevices()
{
	BlockSize.x = 200;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = (int)((N+1)/BlockSize.x) + 1; //    N/BlockSize.x = # of blocks needed, since its int division add 1 to round up
	GridSize.y = 1;
	GridSize.z = 1;
}

// Allocating the memory we will be using.
void allocateMemory()
{	
	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
	
	// Device "GPU" Memory
	cudaMalloc(&A_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU,N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
}

// Loading values into the vectors that we will add.
void innitialize()
{
	for(int i = 0; i < N; i++)
	{		
		A_CPU[i] = (float)i;	
		B_CPU[i] = (float)(3*i);
	}
}

// Adding vectors a and b on the CPU then stores result in vector c.
void dotProductCPU(float *a, float *b, float *C_CPU, int n)
{
	for(int id = 0; id < n; id++)
	{ 
		C_CPU[id] = a[id] * b[id];
	}
	
	for(int id = 1; id < n; id++)
	{ 
		C_CPU[0] += C_CPU[id];
	}
}

// This is the kernel. It is the function that will run on the GPU.
// It adds vectors a and b on the GPU then stores result in vector c.
__global__ void dotProductGPU(float *a, float *b, float *c, int n)
{
    extern __shared__ float C_Shared[]; // Shared memory for partial results
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    // Load data into shared memory
    if (id < n)
    {
        C_Shared[tid] = a[id] * b[id]; //shared memory is only in a single block
		//so global memory is multiplied then stored in shared memory.
		//meaning block one holds 0-199, block two holds 200-399, and so on.
	}
    else
    {
        C_Shared[tid] = 0.0f; //this will happen at the last block if it has less than 200 elements.
		//make sure it's 0 and not garbage, pretty sure this makes it work out, we'll see
    }
    __syncthreads();

    //now changing this is as easy as swapping the c's with C_Shared and the ids with tids
	//fold by block until one element is left in the block.
	int fold = blockDim.x;
	while(1 < fold)
	{
		if(fold%2 != 0) //
		{
			if(tid == 0 && (fold - 1) < n)
			{
				C_Shared[0] = C_Shared[0] + C_Shared[fold - 1];
			}
			fold = fold - 1;
		}
		fold = fold/2;
		if(tid < fold && (tid + fold) < n)
		{
			C_Shared[tid] = C_Shared[tid] + C_Shared[tid + fold];
		}
		__syncthreads();
	}

    // Store the result in global memory
    if (tid == 0)
    {
        c[blockIdx.x] = C_Shared[0]; //each block should have the final result in C_Shared[0]
		//so copy that bad mamma jamma and send it to it's corresponding block index of C_GPU
		//now time to go change the CPU code to do the final reduction after we copy the memory
    }
}

// Checking to see if anything went wrong in the vector addition.
bool check(float cpuAnswer, float gpuAnswer, float tolerence)
{
	double percentError;
	
	percentError = abs((gpuAnswer - cpuAnswer)/(cpuAnswer))*100.0;
	printf("\n\n percent error = %lf\n", percentError);
	
	if(percentError < Tolerance) 
	{
		return(true);
	}
	else 
	{
		return(false);
	}
}

// Calculating elasped time.
long elaspedTime(struct timeval start, struct timeval end)
{
	// tv_sec = number of seconds past the Unix epoch 01/01/1970
	// tv_usec = number of microseconds past the current second.
	
	long startTime = start.tv_sec * 1000000 + start.tv_usec; // In microseconds.
	long endTime = end.tv_sec * 1000000 + end.tv_usec; // In microseconds

	// Returning the total time elasped in microseconds
	return endTime - startTime;
}

// Cleaning up memory after we are finished.
void CleanUp()
{
	// Freeing host "CPU" memory.
	free(A_CPU); 
	free(B_CPU); 
	free(C_CPU);
	
	cudaFree(A_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C_GPU);
	cudaErrorCheck(__FILE__, __LINE__);
}

int main()
{
	timeval start, end;
	long timeCPU, timeGPU;
	//float localC_CPU, localC_GPU;
	
	// Setting up the GPU
	setUpDevices();
	
	// Allocating the memory you will need.
	allocateMemory();
	
	// Putting values in the vectors.
	innitialize();
	
	// Adding on the CPU
	gettimeofday(&start, NULL);
	dotProductCPU(A_CPU, B_CPU, C_CPU, N);
	DotCPU = C_CPU[0];
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	
	// if(BlockSize.x < N)
	// {
	// 	printf("\n\n Your vector size is larger than the block size.");
	// 	printf("\n Because we are only using one block this will not work.");
	// 	printf("\n Good Bye.\n\n");
	// 	exit(0);
	// }
	
	// Adding on the GPU
	gettimeofday(&start, NULL);
	
	// Copy Memory from CPU to GPU		
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	dotProductGPU<<<GridSize,BlockSize, BlockSize *sizeof(float)>>>(A_GPU, B_GPU, C_GPU, N);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Copy Memory from GPU to CPU	
	cudaMemcpyAsync(C_CPU, C_GPU, GridSize.x*sizeof(float), cudaMemcpyDeviceToHost); //copy each block's result to CPU memory
	cudaErrorCheck(__FILE__, __LINE__);

	// do the final reduction on the CPU
	DotGPU = 0.0;
	for(int i = 0; i < GridSize.x; i++)
	{
		DotGPU += C_CPU[i];
	}

	// Making sure the GPU and CPU wiat until each other are at the same place.
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);

	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Checking to see if all went correctly.
	if(check(DotCPU, DotGPU, Tolerance) == false)
	{
		printf("\n\n Something went wrong in the GPU dot product.\n");

		printf("\n DotCPU = %f", DotCPU);
		printf("\n DotGPU = %f", DotGPU);

	}
	else
	{
		printf("\n\n You did a dot product correctly on the GPU");
		printf("\n The time it took on the CPU was %ld microseconds", timeCPU);
		printf("\n The time it took on the GPU was %ld microseconds", timeGPU);
	}
	
	// Your done so cleanup your room.	
	CleanUp();	
	
	// Making sure it flushes out anything in the print buffer.
	printf("\n\n");
	
	return(0);
}


