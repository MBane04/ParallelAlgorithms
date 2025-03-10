// Name: Mason Bane
// Setting up a stream
// nvcc HW15.cu -o temp

/*
 What to do:
 Read chapter 10 in your book and remove all the ??? in the code to see how to setup streams.
*/


// Include files
#include <sys/time.h>
#include <stdio.h>

// Include files
#define DATA_CHUNKS (1024*1024)  //chunks we want to do at a time?
#define ENTIRE_DATA_SET (20*DATA_CHUNKS) //self explanatory
#define MAX_RANDOM_NUMBER 1000
#define BLOCK_SIZE 256

//Globals
dim3 BlockSize; //This variable will hold the Dimensions of your block
dim3 GridSize; //This variable will hold the Dimensions of your grid
float *NumbersOnGPU, *PageableNumbersOnCPU, *PageLockedNumbersOnCPU;
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
cudaEvent_t StartEvent, StopEvent;
//define the stream
cudaStream_t Stream0;

//Function prototypes
void cudaErrorCheck(const char*, int);
void setUpCudaDevices();
void allocateMemory();
void loadData();
void cleanUp();
__global__ void trigAdditionGPU(float *, float *, float *, int );

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


//This will be the layout of the parallel space we will be using.
void setUpCudaDevices()
{
	cudaEventCreate(&StartEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventCreate(&StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaDeviceProp prop;
	int whichDevice;
	
	cudaGetDevice(&whichDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaGetDeviceProperties(&prop, whichDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	if(prop.deviceOverlap != 1)
	{
		printf("\n GPU will not handle overlaps so no speedup from streams");
		printf("\n Good bye.");
		exit(0);
	}
	
	//create the stream
	cudaStreamCreate(&Stream0);
	cudaErrorCheck(__FILE__, __LINE__);
	
	BlockSize.x = BLOCK_SIZE;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	if(DATA_CHUNKS%BLOCK_SIZE != 0)
	{
		printf("\n Data chunks do not divide evenly by block size, sooo this program will not work.");
		printf("\n Good bye.");
		exit(0);
	}
	GridSize.x = DATA_CHUNKS/BLOCK_SIZE;
	GridSize.y = 1;
	GridSize.z = 1;	
}

//Sets a side memory on the GPU and CPU for our use.
void allocateMemory()
{	
	//Allocate Device (GPU) Memory
	cudaMalloc(&A_GPU,DATA_CHUNKS*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,DATA_CHUNKS*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU,DATA_CHUNKS*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	
	//Allocate page locked Host (CPU) Memory
	cudaHostAlloc(&A_CPU, ENTIRE_DATA_SET*sizeof(float), cudaHostAllocDefault);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaHostAlloc(&B_CPU, ENTIRE_DATA_SET*sizeof(float), cudaHostAllocDefault);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaHostAlloc(&C_CPU, ENTIRE_DATA_SET*sizeof(float), cudaHostAllocDefault);
	cudaErrorCheck(__FILE__, __LINE__);
}

void loadData()
{
	time_t t;
	srand((unsigned) time(&t));
	
	for(int i = 0; i < ENTIRE_DATA_SET; i++)
	{		
		A_CPU[i] = MAX_RANDOM_NUMBER*rand()/RAND_MAX;
		B_CPU[i] = MAX_RANDOM_NUMBER*rand()/RAND_MAX;	
	}
}

//Cleaning up memory after we are finished.
void cleanUp()
{
	cudaFree(A_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(B_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFree(C_GPU); 
	cudaErrorCheck(__FILE__, __LINE__);
	
	//free the memory with cudaFreeHost
	cudaFreeHost(A_CPU);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFreeHost(B_CPU);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaFreeHost(C_CPU);
	cudaErrorCheck(__FILE__, __LINE__);
	
	cudaEventDestroy(StartEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventDestroy(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// kill the stream >:)
	cudaStreamDestroy(Stream0);
	cudaErrorCheck(__FILE__, __LINE__);
}

__global__ void trigAdditionGPU(float *a, float *b, float *c, int n)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	
	if(id < n)
	{
		c[id] = sin(a[id]) + cos(b[id]);
	}
}

int main()
{
	float timeEvent;
	
	setUpCudaDevices();
	allocateMemory();
	loadData();
	
	cudaEventRecord(StartEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	
	for(int i = 0; i < ENTIRE_DATA_SET; i += DATA_CHUNKS) //so from my understanding, is this copying the data/'queuing it' as the book said while the GPU is working on the previous data?
	{
		//copy the data to the GPU
		//need 5 args now, the 5th arg is the stream
		cudaMemcpyAsync(A_GPU, A_CPU + i, DATA_CHUNKS*sizeof(float), cudaMemcpyHostToDevice, Stream0); //copy what chunk we are on to the GPU, so jump i amount of data every time
		cudaErrorCheck(__FILE__, __LINE__);
		cudaMemcpyAsync(B_GPU, B_CPU + i, DATA_CHUNKS*sizeof(float), cudaMemcpyHostToDevice, Stream0); //same here
		cudaErrorCheck(__FILE__, __LINE__);
		
		//call the kernel
		trigAdditionGPU<<<GridSize, BlockSize, 0, Stream0>>>(A_GPU, B_GPU, C_GPU, DATA_CHUNKS);
		cudaErrorCheck(__FILE__, __LINE__);
		
		//copy the chunk of data we just did back to the CPU
		cudaMemcpyAsync(C_CPU+i, C_GPU, DATA_CHUNKS*sizeof(float), cudaMemcpyDeviceToHost, Stream0);
		cudaErrorCheck(__FILE__, __LINE__);
	}
	
	//make the CPU wait until the GPU has finished stream0
	cudaStreamSynchronize(Stream0); 
	
	cudaEventRecord(StopEvent, 0);
	cudaErrorCheck(__FILE__, __LINE__);
	// Make the CPU wiat until this event finishes so the timing will be correct.
	cudaEventSynchronize(StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaEventElapsedTime(&timeEvent, StartEvent, StopEvent);
	cudaErrorCheck(__FILE__, __LINE__);
	printf("\n Time on GPU = %3.1f milliseconds", timeEvent);
	
	
	printf("\n");
	//You're done so cleanup your mess.
	cleanUp();	
	
	return(0);
}
