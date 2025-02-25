// Name: Mason Bane
// Vector Dot product on many block 
// nvcc HW10.cu -o temp
/*
 What to do:
 This code is the solution to HW9. It computes the dot product of vectors of any length and uses shared memory to 
 reduce the number of calls to global memory. However, because blocks can't sync, it must perform the final reduction 
 on the CPU. 
 To make this code a little less complicated on the GPU let do some pregame stuff and use atomic adds.
 1. Make sure the number of threads on a block are a power of 2 so we don't have to see if the fold is going to be
    even. Because if it is not even we had to add the last element to the first reduce the fold by 1 and then fold. 
    If it is not even tell your client what is wrong and exit.---------------DONE YAYAYAYAYAYAYAYAYAYAYAYAYAYAYAYAY

 2. Find the right number of blocks to finish the job. But, it is possible that the grid demention is too big. I know
    it is a large number but it is finite. So use device properties to see if the grid is too big for the machine 
    you are on and while you are at it make sure the blocks are not to big too. Maybe you wrote the code on a new GPU 
    but your client is using an old GPU. Check both and if either is out of bound report it to your client then kindly
    exit the program.		---------------DONE

 3. Always check to see if you have threads working past your vector is a real pain and adds a bunch of time consumming
    if statments to your GPU code. To get around this findout how much you would have to add to your vector to make it 
    perfectly fit in your block and grid layout and pad it with zeros. Multipying zeros and adding zero do nothing to a 
    dot product. If you were luck on HW8 you kind of did this but you just got lucky because most of the time the GPU sets
    everything to zero at start up. But!!!, you don't want to put code out where you are just lucky soooo do a cudaMemset
    so you know everything is zero. Then copy up the now zero values. --DONE

 4. In HW9 we had to do the final add "reduction' on the CPU because we can't sync block. Use atomic add to get around 
    this and finish the job on the GPU. Also you will have to copy this final value down to the CPU with a cudaMemCopy.
    But!!! We are working with floats and atomics with floats can only be done on GPUs with major compute capability 3 
    or higher. Use device properties to check if this is true. And, while you are at it check to see if you have more
    than 1 GPU and if you do select the best GPU based on compute capablity. --DONE

 5. Add any additional bells and whistles to the code that you thing would make the code better and more foolproof. --DONE
	-Checks for 0 as block size
	-Checks for 0 as vector size
	-Checks for negative vector size
	-Checks for negative block size
	-added print statements so the user knows what happened on a case by case basis
*/

// Include files
#include <sys/time.h>
#include <stdio.h>

// Defines
#define N 967'521// Length of the vector, works with 100'000, 548'973, 5'489'735, 7, upper limit for 1024 threads is 5'494'778 on my machine BUT IT DOESNT WORK FOR 2'987'465
#define BLOCK_SIZE 1024 // Threads in a block, must be even, tried 1024, 512, 32, 16, 8.. works up to 5'489'000 for sure

// Global variables
float *A_CPU, *B_CPU, *C_CPU; //CPU pointers
float *A_GPU, *B_GPU, *C_GPU; //GPU pointers
float DotCPU, DotGPU;
dim3 BlockSize; //This variable will hold the Dimensions of your blocks
dim3 GridSize; //This variable will hold the Dimensions of your grid
float Tolerance = 0.01;
int vectorFit = N;
cudaDeviceProp *DeviceProps;

// Function prototypes
void cudaErrorCheck(const char *, int);
void setUpDevices();
void setup();
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

	//how many gpus
	int numDevices;
	cudaGetDeviceCount(&numDevices);
	cudaErrorCheck(__FILE__, __LINE__);

	// Allocate memory for the array of device properties
	DeviceProps = (cudaDeviceProp*)malloc(numDevices * sizeof(cudaDeviceProp));

	// Get the properties of each device
	for (int i = 0; i < numDevices; i++)
	{
		cudaGetDeviceProperties(&DeviceProps[i], i);
	}

	// Find the device with the highest compute capability and select the best one, i also added a case for no CUDA capable device
	if(numDevices == 0)
	{
		printf("\n\n There are no available device(s) that support CUDA.\n");
		exit(0);
	}
	
	int bestDevice = 0;
	if(numDevices > 1) //if there is more than 1 device
	{
		
		for (int i = 1; i < numDevices; i++) //compare all devices
		{
			if (DeviceProps[i].major > DeviceProps[bestDevice].major) //if there is a better major	
			{
				bestDevice = i; //i is the better device
			}
			else if (DeviceProps[i].major == DeviceProps[bestDevice].major && DeviceProps[i].minor > DeviceProps[bestDevice].minor) //they tied on major and 1 is better on minor
			{
				bestDevice = i; //i is the better device
			}
			//afaik if neither of those conditions are met then 0 must be the best device
		}
		cudaSetDevice(bestDevice); //apparently there is a function to choose what device you want to be active, cool
		cudaErrorCheck(__FILE__, __LINE__); // just in case
		printf("multiple devices detected, using best device: %s\n", DeviceProps[bestDevice].name);
	}


	//now lets see if we CAN do an atomic add on our best device
	if(DeviceProps[bestDevice].major < 3)
	{
		printf("\n\n This GPU does not support atomicAdd for floats.\n");
		printf("Sucks to suck..... later\n");
		exit(0);
	}

	//if BLOCK_SIZE is not a power of 2
	//also checks if block size is 0 or negative, further idiot proofing the code lol
	if(BLOCK_SIZE > 0 && BLOCK_SIZE & (BLOCK_SIZE -1) != 0) //bitwise AND should always be 0 if a power of 2 is anded with itself - 1 i.e 0010 & 0001 = 0, 0100 & 0011 = 0
	{
		printf("\n\n The number of threads in a block must be a power of 2.\n");
		printf("How about you read the instructions next time you absolute donut....\n");
		printf("\n\n.... you know what? Just don't even try again. I'm outta here.\n\n");
		exit(0);
	}


	if(DeviceProps[bestDevice].maxThreadsPerBlock < BLOCK_SIZE)
	{
		printf("\n\n The number of threads in a block is too big for this GPU.\n");
		printf("Please fix it and try again. :) \n");
		exit(0);
	}

	int gridsNeeded = (int)((N - 1)/BLOCK_SIZE) + 1;
	
	if(gridsNeeded > DeviceProps[bestDevice].maxGridSize[0]) //fits multidimensional grid
	{
		printf("\n\n The number of blocks in a grid is too big for this GPU.\n");
		printf("# of blocks needed = %d\n", gridsNeeded);
		printf("# of blocks allowed = %d\n", DeviceProps[bestDevice].maxGridSize[0]*DeviceProps[bestDevice].maxGridSize[1]*DeviceProps[bestDevice].maxGridSize[2]);
		printf("Please fix it and try again. :) \n");
		exit(0);
	}

	BlockSize.x = BLOCK_SIZE;
	BlockSize.y = 1;
	BlockSize.z = 1;
	
	GridSize.x = gridsNeeded; // This gives us the correct number of blocks.
	GridSize.y = 1;
	GridSize.z = 1;
}

// setup.... allocate the memory you will be using and set it to zero.
void setup()
{
	//first check if the vector fits in the grid, and see how much we need to pad it
	//also checked if vector is positive
	if(N > 0 && N % GridSize.x != 0) //if the vector doesnt fit
	{
		vectorFit = GridSize.x*BlockSize.x; //How many blocks do we need *  how many threads in each block = how big we need to be for a perfect fit

		printf("Grid fitted vector to %d elements.\n", vectorFit);
	}

	// Host "CPU" memory.				
	A_CPU = (float*)malloc(N*sizeof(float));
	B_CPU = (float*)malloc(N*sizeof(float));
	C_CPU = (float*)malloc(N*sizeof(float));
	
	// Device "GPU" Memory
	cudaMalloc(&A_GPU,vectorFit*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&B_GPU,vectorFit*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&C_GPU,sizeof(float));//only need 1 float for the answer
	cudaErrorCheck(__FILE__, __LINE__);

	//set all GPU memory to zero
	cudaMemset(A_GPU, 0, vectorFit*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemset(B_GPU, 0, vectorFit*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemset(C_GPU, 0, sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);

	printf("Memory successfully allocated.\n");

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
	int threadIndex = threadIdx.x;
	int vectorIndex = threadIdx.x + blockDim.x*blockIdx.x;
	__shared__ float c_sh[BLOCK_SIZE];
	
	c_sh[threadIndex] = (a[vectorIndex] * b[vectorIndex]);
	__syncthreads();
	
	int fold = blockDim.x;
	while(1 < fold)
	{
		//no need to check if fold is even because we are using a power of 2 and made sure our vector is a perfect fit
		// if(fold%2 != 0)
		// {
		// 	if(threadIndex == 0 && (vectorIndex + fold - 1) < n)
		// 	{
		// 		c_sh[0] = c_sh[0] + c_sh[0 + fold - 1];
		// 	}
		// 	fold = fold - 1;
		// }
		fold = fold/2;
		if(threadIndex < fold)
		{
			c_sh[threadIndex] = c_sh[threadIndex] + c_sh[threadIndex + fold];
			
		}
		__syncthreads();
	}
	if(threadIndex == 0)
	{
		atomicAdd(c, c_sh[0]);
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

	printf("\n\n\nMemory successfully freed.\n");
	printf("Goodbye.\n");
}

int main()
{
	timeval start, end;
	long timeCPU, timeGPU;
	//float localC_CPU, localC_GPU;
	
	// Setting up the GPU
	setUpDevices();
	
	// Allocating the memory you will need.
	setup();
	
	// Putting values in the vectors.
	innitialize();
	
	// Adding on the CPU
	gettimeofday(&start, NULL);
	dotProductCPU(A_CPU, B_CPU, C_CPU, N);
	DotCPU = C_CPU[0];
	gettimeofday(&end, NULL);
	timeCPU = elaspedTime(start, end);
	
	// Adding on the GPU
	gettimeofday(&start, NULL);
	
	// Copy Memory from CPU to GPU		
	cudaMemcpyAsync(A_GPU, A_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpyAsync(B_GPU, B_CPU, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	
	dotProductGPU<<<GridSize,BlockSize>>>(A_GPU, B_GPU, C_GPU, vectorFit);
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Copy Memory from GPU to CPU	
	cudaMemcpyAsync(&DotGPU, C_GPU, sizeof(float), cudaMemcpyDeviceToHost); //now that we're using atomics we only need to copy 1 float into DotGPU
	cudaErrorCheck(__FILE__, __LINE__);
	
	// Making sure the GPU and CPU wiat until each other are at the same place.
	cudaDeviceSynchronize();
	cudaErrorCheck(__FILE__, __LINE__);

	gettimeofday(&end, NULL);
	timeGPU = elaspedTime(start, end);
	
	// Checking to see if all went correctly.
	if(check(DotCPU, DotGPU, Tolerance) == false)
	{
		printf("\n\n Something went wrong in the GPU dot product.\n");
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


